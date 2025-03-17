class LearningRate:
    def __init__(
                    self, 
                    type:str = 'YOLOv5ParamSchedulerHook',
                    scheduler_type:str = 'linear',
                    lr_factor:int = 0.01,
                    max_epochs: int = 30,
                    warmup_epochs: int = 3,
                    warmup_bias_lr: float = 0.1,
                    warmup_momentum: float = 0.8,
                    warmup_mim_iter: int = 1000
                    
                    ):


        self.lr         = dict(
                    type = type,
                    scheduler_type = scheduler_type,
                    lr_factor = lr_factor,
                    max_epochs = max_epochs,
                    warmup_epochs = warmup_epochs,
                    warmup_bias_lr = warmup_bias_lr,
                    warmup_momentum = warmup_momentum,
                    warmup_mim_iter = warmup_mim_iter
                            )

    

    def __call__(self):
        return self.lr