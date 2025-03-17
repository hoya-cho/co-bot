import os
import mmcv

from mmengine.config import ConfigDict
from mmengine.config import Config as MMCVConfig

class Config: 
    def __init__(self, model_name:str = 'yolov8_x') -> None:

        if model_name == 'yolov8_x' :
            from .yolov8.yolov8_x_mask_refine_syncbn_fast_8xb16_500e_coco import Configs
        if model_name == 'codetr' :
            from .codetr.co_detr import Configs
        else :
            raise ValueError('Model type that does not exist')

        self._configs = MMCVConfig(ConfigDict(Configs))
        
    def get_configs(self) :
        return self._configs