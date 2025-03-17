class Rule:
    def __init__(
                    self, 
                    runner_type:str      = 'EpochBasedTrainLoop',  #EpochBasedTrainLoop or IterBasedTrainLoop                    max_epochs:int       = 2, 
                    max_epochs:int    = 30,
                    val_interval:int    = 1,
                    dynamic_intervals   = [(27, 1)],
                    checkpoint_interval:int = 1,
                    save_best:str ='auto',
                    max_keep_ckpts:int =10,
                    monitor='coco/bbox_mAP',
                    rule='greater',
                    min_delta=0,
                    patience=7
                    ):

        self.train_cfg = dict(
                                type = 'EpochBasedTrainLoop', 
                                max_epochs = max_epochs,
                                val_interval = val_interval,
                                dynamic_intervals = dynamic_intervals)

        
        self.checkpoint  = dict(
                                type='CheckpointHook', 
                                interval=checkpoint_interval, 
                                save_best=save_best,
                                max_keep_ckpts=max_keep_ckpts)
        

        self.earlystopping  = dict(
                                type='EarlyStoppingHook',
                                monitor=monitor,
                                rule=rule,
                                min_delta=min_delta,
                                patience=patience)

    def __call__(self):
        return {'train_cfg' : self.train_cfg, 'checkpoint' :  self.checkpoint, 'earlystopping' : self.earlystopping }