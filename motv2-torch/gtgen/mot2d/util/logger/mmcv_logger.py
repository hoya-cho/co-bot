import os
import json
import logging
from collections import OrderedDict

from abc import *

from . import Logger

logger_initialized = {}

class MMCVLogger(Logger):
    def __init__(
                self, 
                **kwargs
                ):

        super(MMCVLogger, self).__init__(**kwargs)

    def _get_logger(self,
                    name, 
                    log_file=None, 
                    log_level=logging.INFO, 
                    file_mode='w', 
                    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'   
                    ):
        try:
            from mmcv.utils import get_logger
            logger = get_logger(name = name, log_file = log_file, log_level = log_level, file_mode = file_mode)
        except ImportError:
            raise ImportError(
                'Please run "pip install mmcv" to install mmcv')

        return logger
