class Optimizer:
    def __init__(
                    self, 
                    optim_type:str      = 'SGD',
                    lr:float            = 0.0001, 
                    momentum:float      = 0.937, 
                    weight_decay:float  = 0.0005, 
                    nesterov            = True,
                    batch_size_per_gpu  = 4
                    ):

        
        self.optimizer  = dict(
                            type = optim_type,
                            lr = lr,
                            momentum = momentum,
                            weight_decay = weight_decay,
                            nesterov = nesterov,
                            batch_size_per_gpu = batch_size_per_gpu
                            ) 
        

    def __call__(self):
        return self.optimizer