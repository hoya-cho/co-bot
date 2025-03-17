import copy
import torch
import os

from ctypes import c_bool
from multiprocessing import sharedctypes
from attrdict import AttrDict

from gtgen.utils import misc
from ..gtgen_abstract import *
from .mot2d_abstract import *
from .scheduler import LearningRate, Optimizer, Rule
from mmengine.runner import Runner
from .distribute import train_worker
from ..dist_parallel import distribute_worker, setup

from mmengine.device.utils import get_device
from mmyolo.utils import is_metainfo_lower

import torch.distributed as dist

class HyperParameters:
    def __init__(self,
                 lr:LearningRate = LearningRate(),
                 optimizer:Optimizer = Optimizer(),
                 rule:Rule = Rule()
                ):
        self.lr = lr
        self.optimizer = optimizer
        self.rule = rule

class GtGenMot2dLearn(GtGen2dMultiOfTrackingV2Abstract, GtGenModelLearnAbstract): 
    def __init__(self, 
                 devices:tuple, 
                 batch_size:int,
                 num_workers:int,
                 classes:list,              
                 work_dir:str,
                 model_name='yolov8_x',
                ): 
                 
        super().__init__(devices, work_dir, model_name)

        self.cfg.classes    = classes
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.model_name     = model_name
        self.logger         = logger.setup_logger(
        "train" , folder=work_dir, time_filename=True, filename="train"
        )       
    
        misc.init_torch(train_phase=True) 
    
    def train(
        self,
        train_dataset,
        valid_dataset=None, 
        pretrained_path=None,
        parameters:HyperParameters=HyperParameters(),
        backbone_init: bool = False,
        resume: bool = False,
        save_path: str= None,
        progress_callback=None,
        port=30110
    ) -> dict:
        
        self.logger.info("GtGen2dodLearn.train start")
        
        start_t = time.time()

        cfg = copy.deepcopy(self.cfg)
        
        if pretrained_path:
            if resume:
                cfg.load_from   = None
                cfg.resume = pretrained_path
            else :
                cfg.load_from   = pretrained_path
                cfg.resume = None
                
        #param cfg 값 넣어주기
        if parameters is None:
            pass
        else:   
            #max_epoch 수정
            cfg.train_cfg = parameters.rule.train_cfg
            cfg.default_hooks.param_scheduler.max_epochs = parameters.rule.train_cfg['max_epochs']
            #checkpoint 수정
            cfg.default_hooks.checkpoint = parameters.rule.checkpoint
            cfg.default_hooks.earlystopping = parameters.rule.earlystopping
        
        cfg.train_dataloader.batch_size = self.batch_size
        cfg.optim_wrapper.optimizer.batch_size_per_gpu = self.batch_size
        cfg.train_dataloader.num_workers = self.num_workers

        if len(cfg.gpu_ids) == 1:
            distributed = False
        else:
            distributed = True

        meta = dict()

        meta['CLASSES'] = cfg.train_dataloader.dataset.metainfo.classes
        meta['config'] = cfg

        self.meta = meta

        # log some basic info
        self.logger.info(f'Distributed training: {distributed},')
        self.logger.info(f'Config:\n{cfg.pretty_text}')

        is_metainfo_lower(cfg)        

        success = 'failure'
        
        if distributed :
            assert len(cfg.gpu_ids) > 1, "Need to change from single gpu settings to multiple gpu settings"

            self.logger.info('Using distributed learning')
            self.logger.info(f'GPU Count : {len(cfg.gpu_ids)}')

            ngpus_per_node = len(cfg.gpu_ids)
            is_stopping = sharedctypes.Value(c_bool, False, lock=False)
            
            args = AttrDict({
                        'cfg' : cfg,
                        'meta' : meta, 
                        'is_stopping' : is_stopping,
                        'backbone_init' : backbone_init,
                        'progress_callback' : progress_callback})

            setup(master_port = port)
            distribute_worker(
                is_train_phase=True,
                task_worker=train_worker,
                ngpus_per_node=ngpus_per_node,
                args=args,
                progress_callback=progress_callback
            )
            #progress_callback.set_stop()    

        else:
            # build the runner from config
            if 'runner_type' not in cfg:
                # build the default runner
                cfg.local_rank = 0
                runner = Runner.from_cfg(cfg)
            else:
                # build customized runner from the registry
                # if 'runner_type' is set in the cfg
                runner = RUNNERS.build(cfg)

            #백본 초기화 설정
            if backbone_init:
                runner.backbone_init = True # MMengine runner/runner.py 
            # 콜백 함수 설정. loop를 돌때마다 runtime_info_hook.py 에서 정보를 update stop시 현재 정보 리턴
            if progress_callback:
                runner.progress_callback = progress_callback
            
            # start training
            model = runner.train()
            
            # success = model.success
            # ret = {}
            # ret['success'] = success # True
            # ret['last_epoch'] = runner.message_hub.get_info(max_epochs)
            # ret['save_epoch'] = runner.message_hub.get_info(epoch)
            # ret['save_score'] = runner.message_hub.get_info(best_score_key)
            # ret['exception'] = None
        # except Exception as e:
        #         ret = {}
        #         ret['success'] = success # False
        #         ret['exception'] = str(e)
        
        self.logger.info("GtGenOd2DLearn.train end : total time:{}".format(time.time() - start_t))

    def save_model(self, save_path) -> bool:
        
        best_model_path = os.path.join(self.cfg.work_dir, "best.pth")
        creation_time = 0
               
        if len(glob.glob(os.path.join(self.cfg.work_dir, "best_coco_*"))) > 0:
            for model_path in glob.glob(os.path.join(self.cfg.work_dir, "best_coco_*")):
                if creation_time < os.path.getctime(model_path):
                    creation_time = os.path.getctime(model_path)
                    best_model_path = model_path

        if os.path.isfile(best_model_path):
            self.saved_model_path = best_model_path
        elif os.path.isfile(os.path.join(self.cfg.work_dir, "last_checkpoint")):
            with open(os.path.join(self.cfg.work_dir, "last_checkpoint")) as f:
                self.saved_model_path = f.read().strip()
            #self.saved_model_path = os.path.join(self.cfg.work_dir, "latest.pth")
        else:
            self.logger.error("Best model does not exist in that path")

        try:
            super().save_model(save_path)
        except Exception as ex:
            self.logger.error(f"Exception error for model save: {ex}")
            return False
        return True     
        
            
        
        
        
        



        
        
        

        
        
        
        

       