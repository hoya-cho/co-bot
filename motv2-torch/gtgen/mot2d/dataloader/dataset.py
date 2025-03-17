#from mmdet.datasets import build_dataset ,build_dataloader
from mmengine.runner import Runner
from gtgen.gtgen_abstract import PhaseType
import copy
import os

def _create_dataset(
                config, 
                dataset_type,
                data_root,
                data_path, 
                ann_path,  
                **kwargs):
                
    cfg = config
    
    if dataset_type is PhaseType.train:
        
        cfg.train_dataloader.dataset.data_root = data_root
        cfg.train_dataloader.dataset.ann_file = ann_path
        # {'img': 'ext/komatsu/train/'}',
        data_path_dic = dict()
        data_path_dic['img'] = data_path
        cfg.train_dataloader.dataset.data_prefix = data_path_dic
        cfg.train_dataloader.dataset.metainfo.classes = cfg.classes
        #num_classes
        cfg.model.bbox_head.head_module.num_classes = len(cfg.classes)
        cfg.model.train_cfg.assigner.num_classes = len(cfg.classes)
        return  cfg.train_dataloader

    elif dataset_type is PhaseType.train_eval:
        
        cfg.val_dataloader.dataset.data_root = data_root
        cfg.val_dataloader.dataset.ann_file = ann_path
        # {'img': 'ext/komatsu/train/'}',
        data_path_dic = dict()
        data_path_dic['img'] = data_path
        cfg.val_dataloader.dataset.data_prefix = data_path_dic
        cfg.val_dataloader.dataset.metainfo.classes = cfg.classes
        cfg.val_evaluator.ann_file = os.path.join(data_root,ann_path)
        return  cfg.val_dataloader, cfg.val_evaluator
    
    elif dataset_type is PhaseType.eval:
        
        cfg.test_dataloader.dataset.data_root = data_root
        cfg.test_dataloader.dataset.ann_file = ann_path
        # {'img': 'ext/komatsu/train/'}',
        data_path_dic = dict()
        data_path_dic['img'] = data_path
        cfg.test_dataloader.dataset.data_prefix = data_path_dic
        cfg.test_dataloader.dataset.metainfo.classes = cfg.classes
        cfg.test_evaluator.ann_file = os.path.join(data_root,ann_path)

        return  cfg.test_dataloader,  cfg.val_evaluator
    

    elif dataset_type is PhaseType.inferece:
        
        cfg.test_dataloader.dataset.data_root = data_root
        cfg.test_dataloader.dataset.ann_file = ann_path
        # {'img': 'ext/komatsu/train/'}',
        data_path_dic = dict()
        data_path_dic['img'] = data_path
        cfg.test_dataloader.dataset.data_prefix = data_path_dic
        cfg.test_dataloader.dataset.metainfo.classes = cfg.classes
        
        return  cfg.test_dataloader  


def create_dataset(
                    config, 
                    dataset_type, 
                    data_root:str,
                    data_path:str, 
                    ann_path:str,
                    **kwargs):
    cfg = config
    
    dataset = _create_dataset(cfg, dataset_type, data_root, data_path, ann_path, **kwargs)
    return dataset
