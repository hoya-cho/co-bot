from ..gtgen_abstract import *
from .od2d_abstract import *
from ..dist_parallel import distribute_worker, setup
from .distribute import test_worker
from gtgen.utils import misc, logger
from multiprocessing import sharedctypes, shared_memory

class GtGenOd2dEvaluation(GtGen2dObjectDetectionV2Abstract, GtGenModelEvaluationAbstract):
    def __init__(self,
                 work_dir,
                 devices:tuple,
                 model_name='yolov8_x'
                ):
                 
        super().__init__(devices, work_dir, model_name)
        self.logger         = logger.setup_logger(
        "Eval" , folder=work_dir, time_filename=True, filename="Eval"
        )
        misc.init_torch(train_phase=False)
    
    # for eval, inference
    def load_model(self, load_path) -> bool:
        
        self.cfg = self._load_model(self.model, load_path)   
        return self.cfg
    def evaluation(
                self,
                dataset,
                det_thres=0.35,
                progress_callback=None,
                port = 30010
                ) -> dict:

        cfg = self.cfg
        cfg.model_test_cfg.score_thr = det_thres
        cfg.val_dataloader = dataset    

        distributed = False
        if len(self.gpu_ids) > 1 :
            distributed = True

        if distributed :
            self.logger.info('Using distributed test')
            self.logger.info(f'GPU Count : {len(self.gpu_ids)}')

            ngpus_per_node = len(self.gpu_ids)
            init_list_size = [ None for _ in range(1024)]
            result_shared_list = shared_memory.ShareableList(init_list_size)
            result_shm_name = result_shared_list.shm.name

            args = AttrDict({
                        'cfg' : cfg,
                        'mem_name' : result_shm_name,
                        'progress_callback' : progress_callback})
            
            setup(master_port = port)
            results = distribute_worker(
                is_train_phase=False,
                task_worker=test_worker,
                ngpus_per_node=ngpus_per_node,
                args=args,
                progress_callback=progress_callback
            )
            
            result_list = list(result_shared_list)
            last_char_index = list(result_shared_list).index(None)
            # #results = eval(''.join(result_list[:last_char_index]))
            results = ''.join(result_list[:last_char_index])
            
            result_shared_list.shm.close()
            result_shared_list.shm.unlink()
            
            return results

        else:
            if 'runner_type' not in cfg:
            # build the default runner
                cfg.local_rank = 0
                runner = Runner.from_cfg(cfg)
            else:
                # build customized runner from the registry
                # if 'runner_type' is set in the cfg
                runner = RUNNERS.build(cfg)

            # start testing
            result = runner.test()

            return result
            