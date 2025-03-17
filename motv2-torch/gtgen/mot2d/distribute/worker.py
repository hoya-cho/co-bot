import os
import random
import warnings
import copy
import numpy as np
import torch

from torch import distributed as dist

from multiprocessing import shared_memory
#from mmcv.parallel import MMDistributedDataParallel
from mmengine.model.wrappers import MMDistributedDataParallel
#from mmcv.runner import build_optimizer, build_runner
#from mmengine.runner import build_param_scheduler
from mmengine.registry import Registry, build_runner_from_cfg
from mmengine.runner import Runner

#from ..mmdet.apis import train_detector, multi_gpu_test
#from ..dataloader import create_dataloader
from gtgen.dist_parallel import get_child_progress_callback
from gtgen.utils import logger

MAX_RESULT_STR = 2048

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_worker(rank, ngpus_per_node, args):    
    
    cfg                    =   args.cfg
    meta                   =   args.meta
   
    log_postfix = "_" + str(rank)
    # run_logger = logger.setup_logger(
    #     "Train" + log_postfix, folder=cfg.work_dir, time_filename=True, filename="train" + log_postfix
    # )

    try:
        progress_callback      =   get_child_progress_callback(True, rank, args, args.progress_callback)
        is_stopping            =   args.is_stopping
        
        gpu = cfg.gpu_ids[rank]  
        
        torch.cuda.set_device(gpu)
        dist.init_process_group(
                                backend = 'nccl', 
                                world_size = ngpus_per_node, 
                                rank = rank)

        cfg.launcher = 'pytorch'

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            cfg.local_rank = gpu
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        #백본 초기화 설정
        if args.backbone_init:
            runner.backbone_init = True # MMengine runner/runner.py 
    #     # 콜백 함수 설정. loop를 돌때마다 runtime_info_hook.py 에서 정보를 update stop시 현재 정보 리턴
        if progress_callback:
            runner.progress_callback = progress_callback
         
        # start training

        model = runner.train()

    except Exception as e:
        logger.write_exception_log(e, " train_worker failure")
        
    progress_callback.end_progress()
    dist.destroy_process_group()


def test_worker(rank, ngpus_per_node, args):
    
    cfg                    =   args.cfg
    #model                  =   args.model
    #datasets               =   args.datasets
    #gpu                    =   cfg.gpu_ids[rank]  
    mem_name               =   args.mem_name
    #logger                 =   args.logger
    progress_callback      =   args.progress_callback
    
    progress_callback   =   get_child_progress_callback(True, rank, args, progress_callback)

    gpu = cfg.gpu_ids[rank]  
        
    torch.cuda.set_device(gpu)
    dist.init_process_group(
                                backend = 'nccl', 
                                world_size = ngpus_per_node, 
                                rank = rank)
        
        
    cfg.launcher = 'pytorch'

    if 'runner_type' not in cfg:
        # build the default runner
        cfg.local_rank = gpu
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

        # start testing
    results = runner.test()
    
    if rank == 0 :
        result_shared_list = shared_memory.ShareableList(name = mem_name)
        eval_list = list(str(results))
        for idx, eval_char in enumerate(eval_list) :
                result_shared_list[idx] = eval_char
                
        result_shared_list.shm.close()
    
    progress_callback.end_progress()