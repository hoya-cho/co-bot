from abc import *

class LoggerBase(metaclass=ABCMeta):
   
    @abstractmethod
    def update_buffer(self, vars):     
        pass

    @abstractmethod
    def _get_logger(self,
                    name, 
                    log_file, 
                    log_level, 
                    file_mode, 
                    format   
                    ):
        pass

    @abstractmethod
    def print_log(self, msg, level):
        pass
        