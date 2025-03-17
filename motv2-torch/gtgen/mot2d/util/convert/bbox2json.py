import os
import hashlib
import cv2
import json


class InfBase:
    def __init__(self, classes:tuple, expt_cls:tuple={}):
        #self.include_score = inf_config.include_score
        self.expt_cls = expt_cls
        self.cls_map =  classes

    def get_annos(result, classes):
        annotations = list()
        #src_encoded = filename.encode() if isinstance(filename, str) else filename.tobytes()
        
        for anno_id, instancedata in enumerate(result, 1):
            
            left, bottom, right, top = map(round,instancedata.bboxes.tolist()[0])
            #print(left, bottom, right, top)
            
            cls_name = classes[int(instancedata.labels)]
            #print(cls_name)
            score = float(instancedata.scores)
            #print(score)
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

    def get_json(result, filename, classes):
        json_result = dict()
        parent_path, filename = os.path.split(filename)
        json_result['filename'] = filename
        json_result['parent_path'] = parent_path
        json_result['attributes'] = dict()
        json_result['annotations'] = get_annos(result=result, classes=classes)
        return json_result
        
    '''
    def save_json(self, result, src, base_dir, dst_base_dir, empty_result_save=False):
        json_result = self.get_json(result=result, src=src, base_dir=base_dir)
        
        # empty_result_save == False, then don't save json file
        if not (empty_result_save or json_result['annotations']):
            return False
        
        parent_path = json_result['parent_path'][1:]  # Remove first string (os.sep)
        #dst_dir = os.path.join(dst_base_dir, parent_path)
        dst_dir = dst_base_dir
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        dst = os.path.join(dst_dir, FileManager(path=src).fname + '.json')
        FileManager.json_dump(obj=json_result, dst=dst)
        return True
    '''

class FileManager:
    def __init__(self, path):
        self.path = path
        self._get_path_info(path=path)

    def _get_path_info(self, path):
        path_without_ext, ext = os.path.splitext(path)
        basename_without_ext = os.path.basename(path_without_ext)
        self.dir_path = os.path.dirname(path)
        self.basename_without_ext = basename_without_ext
        self.ext = ext
        self.fname = basename_without_ext + ext
        
    def get_parent_path(self, base_dir):
        base_dir = base_dir[:-1] if base_dir.endswith(os.sep) else base_dir
        base_dir_depth = len(base_dir.split(os.sep))
        parent_path = self.dir_path.split(os.sep)[base_dir_depth:]
        parent_path = os.path.join(*parent_path) if parent_path else ''
        parent_path = os.path.join(os.sep, parent_path)
        return parent_path

    @staticmethod
    def get_all_fpaths(src_dir, exts=['.png', '.jpg', '.jpeg']):
        fpaths = list()
        for current_path, dirnames, fnames in os.walk(src_dir, followlinks=True):
            if not fnames:
                continue
                
            for fname in fnames:
                fname_ext = FileManager(path=fname).ext
                if not fname_ext in exts:
                    continue
                fpath = os.path.join(current_path, fname)
                fpaths.append(fpath)
        return fpaths

    @staticmethod
    def json_dump(obj, dst):
        with open(dst, 'wt', encoding='utf-8') as j:
            json.dump(obj, j, indent=4, ensure_ascii=False)