import os
import cv2
import random
import numpy as np

from gtgen.utils import misc, logger
#from .evaluation import *
from mmdet.apis import inference_detector, init_detector
try:
    from sahi.slicing import slice_image
except ImportError:
    raise ImportError('Please run "pip install -U sahi" '
                      'to install sahi first for large image inference.')
from mmdet.utils.misc import get_file_list
from mmdet.utils.large_image import merge_results_by_nms, shift_predictions
import torch
import copy


class GtGenMot2dInference(GtGen2dMultiOfTrackingV2Abstract, GtGenModelInferenceAbstract):
    def __init__(
                self,
                work_dir,
                devices: tuple,
                use_sahi = True,
                sahi_max_inst_size =33,
		        sahi_manual_patch_size = None,
                model_name='yolov8_x'): 
                
                
        super().__init__(devices, work_dir, model_name)
        misc.init_torch(train_phase=False)
        self.use_sahi = use_sahi
        self.sahi_max_inst_size = sahi_max_inst_size
        if sahi_manual_patch_size is not None :
            self.sahi_manual_patch_size = sahi_manual_patch_size
            self.auto_slice_resolution = False
        else:
            self.sahi_manual_patch_size = None
            self.auto_slice_resolution = True
        
     # for eval, inference # 체크 포인트에서 모델네임 가져와서 model_cfg 가져오게 수정.
    def load_model(self, load_path) -> bool:
        self.cfg = self._load_model(self.model, load_path) 
        
        return self.cfg
    
    def inference(self,
                  parent_path,
                  inputs_path_list, # parent_path, input_path_list 로 변경
                  score_thr=0.3
                ):
        cfg = self.cfg
        
        cfg.test_dataloader.dataset.data_root = parent_path
        data_path_dic = dict()
        data_path_dic['img'] = inputs_path_list
        cfg.test_dataloader.dataset.data_prefix = data_path_dic
        cfg.test_dataloader.dataset.metainfo.classes = cfg.classes
        
        if not self.use_sahi: #inference_normal 다른곳에 구현
               
            model = init_detector(
            cfg, cfg.load_from, device='cuda:0', cfg_options={})
            files, source_type = get_file_list(os.path.join(parent_path,inputs_path_list))
            dataset_classes = model.dataset_meta.get('classes')
            self.classes = dataset_classes
            
            result_list = []
            for file in files:
                result = inference_detector(model, file)
                #pred_instances = result.pred_instances
                aimmo_gt = get_json(result, file, self.classes , score_thr)
                result_list.append(aimmo_gt)
            return result_list
        
        else: #inference_sahi
                
            model = init_detector(
            cfg, cfg.load_from, device='cuda:0', cfg_options={})
            files, source_type = get_file_list(os.path.join(parent_path,inputs_path_list))
            dataset_classes = model.dataset_meta.get('classes')
            self.classes = dataset_classes
            result_list=[]
            for file in files:
                
                img = mmcv.imread(file)

                # arrange slices
                height, width = img.shape[:2]
                sliced_image_object, slice_width, slice_height = slice_image(
                    img,
                    slice_height=self.sahi_manual_patch_size,
                    slice_width=self.sahi_manual_patch_size,
                    auto_slice_resolution=self.auto_slice_resolution,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                )

                slice_results = []
                start = 0
                while True:
                    # prepare batch slices
                    end = min(start + 1, len(sliced_image_object))
                    images = []
                    for sliced_image in sliced_image_object.images[start:end]:
                        images.append(sliced_image)

                    #크기 필터링
                    result = inference_detector(model, images)
                    result = result[0]
                    
                    size_filtering_idx=[] 
                    for i in range(len(result.pred_instances.bboxes)):
                        # 인스턴스 크기 계산
                        area = abs((result.pred_instances.bboxes[i][0]-result.pred_instances.bboxes[i][2]) * (result.pred_instances.bboxes[i][1]-result.pred_instances.bboxes[i][3]))
                        if area < self.sahi_max_inst_size*self.sahi_max_inst_size: #coco medium size 아래인 것들만 검출, 인스턴스 사이즈를 옵션으로! 
                            size_filtering_idx.append(i) 
                    new_bboxes = torch.zeros(len(size_filtering_idx),4).to("cuda:0")
                    new_scores = torch.zeros(len(size_filtering_idx)).to("cuda:0")
                    new_labels = torch.zeros(len(size_filtering_idx)).to("cuda:0")
                    
                    idx=0
                    for j in size_filtering_idx:
                        new_bboxes[idx] = result.pred_instances.bboxes[j]
                        new_scores[idx] = result.pred_instances.scores[j]
                        new_labels[idx] = result.pred_instances.labels[j]
                        idx+=1
                    
                    result.pred_instances.pop('bboxes', None) 
                    result.pred_instances.pop('scores', None) 
                    result.pred_instances.pop('labels', None) 
                    
                    result.pred_instances.bboxes = new_bboxes
                    result.pred_instances.scores = new_scores
                    result.pred_instances.labels = new_labels.int()
                    
                    # forward the model
                    slice_results.extend([result])
                    if end >= len(sliced_image_object):
                        break
                    start += 1

                slice_results.extend([inference_detector(model, img)])
                
                new_starting_pixels = copy.copy(sliced_image_object.starting_pixels)
                
                new_starting_pixels.append([0,0])
                
                image_result = merge_results_by_nms(
                    slice_results,
                    new_starting_pixels,
                    src_image_shape=(height, width),
                    nms_cfg={
                        'type': 'greedy_nmm',
                        'iou_threshold': 0.25
                })
                aimmo_gt = get_json_sahi(image_result, file, self.classes, score_thr)
                result_list.append(aimmo_gt)
            return result_list        

    # def make_aimmoGT( # 
    #                 self, 
    #                 result, 
    #                 filename:str,
    #                 score_thr:float = 0.3 ,
    #                 expt_cls:list = [],
    #                 with_bbox:bool = False):
        
    #     aimmo_gt_list=[]
        
    #     if not self.use_sahi:
    #         for re in result:
    #             aimmo_gt = get_json(result, filename, self.classes , score_thr)
    #             aimmo_gt_list.append(aimmo_gt)
    #         return aimmo_gt_list
    #     else:
    #         for re in result:
    #             aimmo_gt = get_json_sahi(result, filename, self.classes, score_thr)
    #             aimmo_gt_list.append(aimmo_gt)
    #         return aimmo_gt_list        
   
def get_annos(result, classes, score_thr=0.3):
    annotations = list()
    #src_encoded = filename.encode() if isinstance(filename, str) else filename.tobytes()
    result = result.pred_instances[result.pred_instances.scores > score_thr]
    for anno_id, instancedata in enumerate(result, 1):
        
        left, bottom, right, top = map(round,instancedata.bboxes.tolist()[0])
        
        cls_name = classes[int(instancedata.labels)]
        score = float(instancedata.scores)
    
        import uuid
        
        anno = dict()
        anno['id'] = str(anno_id)+'-'+str(uuid.uuid1())
        anno['type'] = 'bbox'
        anno['points'] = [[left, top], [right, top], [right, bottom], [left, bottom]]
        anno['label'] = cls_name
        anno['attributes'] = dict()
        anno['score'] = score
                        
        annotations.append(anno)

    return annotations

def get_json(result, filename, classes, score_thr):
    json_result = dict()
    parent_path, filename = os.path.split(filename)
    json_result['filename'] = filename
    json_result['parent_path'] = parent_path
    json_result['attributes'] = dict()
    json_result['annotations'] = get_annos(result=result, classes=classes, score_thr=score_thr )

    return json_result

def get_annos_sahi(result, score_thr, classes):
    annotations = list()
    #src_encoded = filename.encode() if isinstance(filename, str) else filename.tobytes()
    new_idx=[]
    try:   
        result.pred_instances.scores = result.pred_instances.scores.tolist()
        
        for i in range(len(result.pred_instances.scores)):
            
            if result.pred_instances.scores[i] > score_thr:
                new_idx.append(i)    
    
        for j in new_idx:
        
            left, bottom, right, top = map(round, result.pred_instances.bboxes[j].tolist())
            cls_name = classes[result.pred_instances.labels[j]]
            score = float(result.pred_instances.scores[j])
            
            import uuid
            
            anno = dict()
            anno['id'] = str(j+1)+'-'+str(uuid.uuid1())
            anno['type'] = 'bbox'
            anno['points'] = [[left, top], [right, top], [right, bottom], [left, bottom]]
            anno['label'] = cls_name
            anno['attributes'] = dict()
            anno['score'] = score
                            
            annotations.append(anno)
    except:
        pass
    return annotations    
        

def get_json_sahi(result, filename, classes, score_thr=0.3):
    json_result = dict()
    parent_path, filename = os.path.split(filename)
    json_result['filename'] = filename
    json_result['parent_path'] = parent_path
    json_result['attributes'] = dict()
    json_result['annotations'] = get_annos_sahi(result=result, score_thr=score_thr, classes=classes)
    return json_result