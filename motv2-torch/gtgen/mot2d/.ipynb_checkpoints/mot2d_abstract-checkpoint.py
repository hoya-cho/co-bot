import os
import json
import torch
import mmcv
import time
import logging

from attrdict import AttrDict

from ..gtgen_abstract import PhaseType
from .config import Config

from .util import load_checkpoint, get_state_dict, weights_to_cpu
from gtgen.utils import logger
from mmengine import Registry, build_model_from_cfg
from mmyolo.utils import is_metainfo_lower

from mmengine.runner import Runner

class GtGen2dMultiOfTrackingV2Abstract: 
    def __init__(
                self,
                devices:list,
                work_dir,
                model_name,
                ):
        self.cfg = Config(model_name = model_name).get_configs()
        self.logger = None
        self.gpu_ids = devices
        self.cfg.gpus = len(devices)
        self.cfg.work_dir = work_dir
        self.cfg.gpu_ids = devices
        '''
        self.distributer = DistributeWorker(master_port = '30000', world_size = len(devices))
        '''
        self.model = None

    def create_dataloader(self,
                          phase:PhaseType,
                          **kwargs):
        pass
    
    def create_dataset(self, dataset_type:PhaseType, data_root, data_path, ann_path, **kwargs) :
        
        from .dataloader import create_dataset as _create_dataset

        self.logger.info("create_dataset start")
        start_t = time.time()            
        dataset = _create_dataset(self.cfg, dataset_type, data_root, data_path, ann_path, **kwargs)

        self.logger.info("create_dataloader.end ")
        return dataset
    
    def has_model(self) -> bool:
        return self.model is not None

    def _assert_no_model(self):
        assert self.model is not None, "no loaded model"
    
    def get_model_config(self) -> dict:
        self._assert_no_model()

        assert hasattr(self.model, 'cfg'), "Configuration does not exist in model properties"

        return self.model.cfg 

    def get_model_meta(self) -> dict :
        self._assert_no_model()
        
        meta = {}
        
        if hasattr(self.model, 'epoch') :
            meta.update(epoch = self.model.epoch)

        if hasattr(self.model, 'iter') :
            meta.update(iter = self.model.iter)

        if hasattr(self.model, 'best_score') :
            meta.update(best_score = self.model.best_score)

        if hasattr(self.model, 'monitor') :
            meta.update(monitor = self.model.monitor)

        return meta

    
    def create_model(self, mode:PhaseType = PhaseType.train, init_cfg=None) :
        
        model_config = self.cfg.model
        
        model = None         
        
        if mode == PhaseType.train :
            MODELS = Registry('models')
            
            model = build_model_from_cfg(model_config, MODELS, train_cfg=self.cfg.get('train_cfg'),
                         test_cfg=self.cfg.get('test_cfg'))            
        else: #테스트시 모델 cfg 확인해보기
            MODELS = Registry('models')
            model = build_model_from_cfg(
                        model_config, MODELS
                        )

        return model

    def _load_model(self, model, checkpoint_path:str):
            
        from gtgen.mot2d.util.checkpoint import _load_checkpoint
        if self.logger == None:
            self.logger         = logger.setup_logger(
        "Inference" , folder=self.cfg.work_dir, time_filename=True, filename="Eval"
        )     
        self.logger.info('Start load model.')

        cfg = self.cfg
        device_id = torch.cuda.current_device()
        
        if checkpoint_path is not None:
            
            cfg.load_from = checkpoint_path
            is_metainfo_lower(cfg)
            
            checkpoint = _load_checkpoint(cfg.load_from, map_location=lambda storage, loc: storage.cuda(device_id))
            
            if not 'classes' in checkpoint['message_hub'].get('runtime_info')['dataset_meta']:
                self.logger.info('CLASSES is None.')
                return None
            
            #cfg.test_dataloader.dataset.metainfo.classes = checkpoint['message_hub'].get('runtime_info')['dataset_meta']['classes']
            cfg.classes = checkpoint['message_hub'].get('runtime_info')['dataset_meta']['classes']
            cfg.model.bbox_head.head_module.num_classes = len(cfg.classes)
            cfg.model.train_cfg.assigner.num_classes = len(cfg.classes)
        else :
            self.logger.info('Checkpoint path is None.')
            return None

        self.logger.info('Finish load model.')

        return cfg
        #return checkpoint

    #제공 X
    def _save_model(self, model, checkpoint_path:str, meta:dict):
        
        self.logger.info('Start save model.')

        self.logger.print_log('Finish save model.')
        return True

    # for MLOps. ex) ("vehicle", "pedestrian")
    def get_model_classes(self) -> tuple:
        self._assert_no_model()
        return self.model.CLASSES
    
    def set_devices(self, devices:list = [0]):
        self.cfg.gpus = len(devices)
        self.cfg.gpu_ids = devices
        self.gpu_ids = devices