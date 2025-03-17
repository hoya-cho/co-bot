# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from botsort.yolox.utils.visualize import plot_tracking

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from botsort.tracker.mc_bot_sort import BoTSORT
from botsort.tracker.tracking_utils.timer import Timer
from mmdet.utils.misc import get_file_list
import numpy as np
import os.path as osp
import os
import time
import torch 
import json

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection img demo')
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args

def extract_files_with_interval(file_list, interval):
    extracted_files = []
    for i in range(0, len(file_list), interval):
        extracted_files.append(file_list[i])
    return extracted_files

def json_dump(obj, dst):
    with open(dst, 'wt', encoding='utf-8') as j:
        json.dump(obj, j, indent=4, ensure_ascii=False)

def main():
    args = parse_args()
    
    break
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    classes_info= model.dataset_meta.get('classes')
    print(classes_info)
    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta
    # get file list
    files, source_type = get_file_list(args.img)
    # video_reader = mmcv.VideoReader(args.video)
    # video_writer = None
    # if args.out:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_writer = cv2.VideoWriter(
    #         args.out, fourcc, video_reader.fps,
    #         (video_reader.width, video_reader.height))
    
    t_args = argparse.Namespace(
        ablation=False, 
        appearance_thresh=0.25, 
        camid=0, 
        ckpt='/ext/hoya/bytetrack_x_mot17.pth.tar', 
        cmc_method='sparseOptFlow', 
        conf=None, demo='video', 
        device='gpu', 
        exp_file='/mmdetection/botsort/yolox/exps/example/mot/yolox_x_mix_det.py', 
        experiment_name='yolox_x_mix_det', 
        fast_reid_config='/mmdetection/botsort/fast_reid/configs/MOT17/sbs_S50.yml', 
        fast_reid_weights='/ext/hoya/mot17_sbs_S50.pth', 
        fp16=True, 
        fps=30, 
        fuse=True, 
        fuse_score=True, 
        match_thresh=0.8, #0.8
        min_box_area=10, 
        mot20=False, 
        name=None, 
        new_track_thresh=0.7, #0.7
        nms=None, 
        path='/ext/mot_sample_video/newyork_sample.mp4', 
        proximity_thresh=0.5, 
        save_result=True, 
        track_buffer=30, 
        track_high_thresh=0.6, 
        track_low_thresh=0.1, 
        trt=False, 
        tsize=None, 
        with_reid=True
    )
    #print(t_args)
    tracker = BoTSORT(t_args)
    # vid_writer = cv2.VideoWriter(
    #     '/bot_sort_test13.mp4', cv2.VideoWriter_fourcc(*"mp4v"), video_reader.fps, (int(video_reader.width), int(video_reader.height))
    # )
    timer = Timer()
    frame_id = 0
    results = []
    files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    interval = 1
    files = extract_files_with_interval(files, interval)
    
    out_dir = '/ext/BMW_tracking_all_every_7pth'
    #filtering_classes = [0,1,2,3,4,6,7] # 7pth vehicle
    #filtering_classes = [3,4,5,8,9,11,12,13]
    filtering_classes = None
    #filtering_classes = [0,2,3,4,5,8,10,11,12,13]
    
    for frame in track_iter_progress((files, len(files))):
        #print(frame, len(frame))
        #print(frame.shape)
        frame_path = frame
        frame = mmcv.imread(frame)
        frame = mmcv.imconvert(frame, 'bgr', 'rgb')
        result = inference_detector(model, frame, test_pipeline=test_pipeline)
        
        pre_instances = result.pred_instances
        #Filtering class
        if filtering_classes is not None:
            indices = torch.isin(result.pred_instances.labels.to('cuda:0'), torch.tensor(filtering_classes).to('cuda:0'))
            #print(indices)
            pre_instances = pre_instances[indices]
        
        #filtering_classes_indices = torch.where(result.pred_instances.labels in filtering_classes)[0]
        #pre_instances = result.pred_instances[result.pred_instances.labels.isin(filtering_classes)]
        #print(pre_instances)
        
        #print(result)
        outputs = np.empty((1,7))
        
        for i, instancedata in enumerate(pre_instances):
            output = np.array([])
            left, bottom, right, top = map(round, instancedata.bboxes.tolist()[0])
            #left,top,right,bottom
            output = np.append(output, left)
            output = np.append(output, bottom)
            output = np.append(output, right)
            output = np.append(output, top)
            score = instancedata.scores.tolist()[0]
            label = instancedata.labels.tolist()[0]
            
            output = np.append(output, score)
            output = np.append(output, 1)
            output = np.append(output, label)
            outputs = np.vstack((outputs,output))
        #print(outputs, type(outputs), len(outputs))
        #print(outputs)
        online_targets = tracker.update(outputs[1:], frame)
    
        #print(online_targets)
        
        online_tlwhs = []
        online_ids = []
        online_scores = []
        classes= model.dataset_meta.get('classes')
        annotations = list()
        for anno_id, t in enumerate(online_targets, 1):
            
            tlwh = t.tlwh
            tid = t.track_id
            cls_name = classes[int(t.label)]
            left = int(tlwh[0])
            if left < 0 :
                left = 0
            bottom = int(tlwh[1]+tlwh[3] )
            if bottom > 1195:
                bottom = 1195
            right = int(tlwh[0]+tlwh[2])
            if right > 3840:
                right = 3840
            top = int(tlwh[1])
            if top < 0:
                top = 0
            
            #if tlwh[2] * tlwh[3] > 10:
            import uuid
            
            anno = dict()
            anno['id'] = str(anno_id)+'-'+str(uuid.uuid1())
            anno['type'] = 'bbox'
            anno['points'] = [[left, top], [right, top], [right, bottom], [left, bottom]]
            anno['label'] = cls_name
            anno['track_id'] = str(tid)
            anno['attributes'] = dict()
            #anno['attributes'] = dict(score=int(round(t.score,1)*10),
            #                            track_id=tid)
            
                    
            annotations.append(anno)
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            results.append(
                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},{cls_name},-1,-1,-1\n"
            )

        json_result = dict()
        parent_path, filename = os.path.split(frame_path)
        json_result['filename'] = filename
        json_result['parent_path'] = parent_path
        json_result['attributes'] = dict()
        json_result['annotations'] = annotations
        
        
        if out_dir == "":
            out_dir = "/ext/hoya/co_detr_aimmo_tracking_result/"
            aimmo_json_path = osp.join(out_dir,'aimmo', json_result['parent_path'][1:], json_result['filename'] + '.json')
            result_img_path = osp.join(out_dir,'img', json_result['parent_path'][1:], json_result['filename'] + '.json.jpg')
        else:
            aimmo_json_path = osp.join(out_dir,'aimmo', json_result['parent_path'][1:], json_result['filename'] + '.json')
            result_img_path = osp.join(out_dir,'img', json_result['parent_path'][1:], json_result['filename'] + '.json.jpg')
        #print(aimmo_json_path)
        if not os.path.exists(osp.join(out_dir,'aimmo', json_result['parent_path'][1:])):
            os.makedirs(osp.join(out_dir,'aimmo', json_result['parent_path'][1:]))
        if not os.path.exists(osp.join(out_dir,'img', json_result['parent_path'][1:])):
            os.makedirs(osp.join(out_dir,'img', json_result['parent_path'][1:]))
        json_dump(json_result, aimmo_json_path)
        
        timer.toc()
        online_im = plot_tracking(
                frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
        )
        # vid_writer.write(online_im)
        #cv2.imwrite(os.path.join('/img_output_bmw_ad/',str(frame_id)+"_frame.jpg"), online_im)
        cv2.imwrite(result_img_path, online_im)
        
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
        frame_id += 1
        #timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        
        
        
    res_file = osp.join("/demo_tracking_0.12.txt")
    with open(res_file, 'w') as f:
        f.writelines(results)       
        
    
        #scale = min(800 / float(ori_shape[0], ), 1440 / float(ori_shape[1]))
        
        # detections = []
        # if outputs[0] is not None:
        #     outputs = outputs[0].cpu().numpy()
        #     detections = outputs[:, :7]
        #     detections[:, :4] /= scale

        
    #     visualizer.add_datasample(
    #         name='video',
    #         image=frame,
    #         data_sample=result,
    #         draw_gt=False,
    #         show=False,
    #         pred_score_thr=args.score_thr)
    #     frame = visualizer.get_image()

    #     if args.show:
    #         cv2.namedWindow('video', 0)
    #         mmcv.imshow(frame, 'video', args.wait_time)
    #     if args.out:
    #         video_writer.write(frame)

    # if video_writer:
    #     video_writer.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
