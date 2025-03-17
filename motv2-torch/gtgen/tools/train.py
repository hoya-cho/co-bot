import json
import argparse
import json
from gtgen.gtgen_abstract import *
from gtgen.od2d import GtGenOd2dLearn

from gtgen.gtgen_abstract import ProgressCallBackAbstract
 
def get_class_map(path):
    return json.load(open(path, "r"))


class ProgressTrainCallBack(ProgressCallBackAbstract):
    def __init__(self):
        self.stopping = False

    def __call__(self, **kwargs):
        
        if self.stopping :
            return self.stopping

        print(f"progress: {kwargs}", flush = True)


def main():

    #class_map = get_class_map(args.cls_map)
    #CLASS = ['Vehicle','Pedestrian']
    CLASS = ('rock',)
    od2d_learner = GtGenOd2dLearn(
                                devices = [0], 
                                batch_size = 2, 
                                num_workers = 2 ,
                                classes = CLASS ,
                                model_name = 'yolov8_x',
                                work_dir = "./work_dir/" )
 
    
    train_dataset = od2d_learner.create_dataset(
        PhaseType.train, 
        "/",
        "ext/komatsu/train", 
        "ext/komatsu/komatsu_train.json"
    )
    val_dataset = od2d_learner.create_dataset(
            PhaseType.train_eval, 
            "/",
            "ext/komatsu/train", 
            "ext/komatsu/komatsu_train.json"
    )

    
    progress_callback = ProgressTrainCallBack()

    ret = od2d_learner.train(train_dataset,
                        val_dataset,
                        pretrained_path = '/ext/hoya/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120411-079ca8d1.pth',
                        parameters = None,
                        
                        backbone_init = True,
                        resume = False,
                        #progress_callback = progress_callback,
                        port=30100)
    
if __name__ == '__main__' :
    main()