import abc
import enum
import shutil

class GtGenException(Exception):
    def __init__(self, msg:str, logger=None):
        super().__init__(str)
        if logger is not None:
            logger.error(str)


class PhaseType(enum.Enum):
    train = 0
    train_eval = 1
    eval = 2
    inferece = 3


class EarlyStopping:
    def __init__(self, patience=10, delta=0.01) -> None:
        self.patience = patience      # if patience <= 0, early stopping will not work
        self.delta = delta


class HyperParameters:
    def __init__(self, lr, optimizer, rule):
        self.lr = lr # float
        self.optimizer = optimizer
        self.rule = rule


class DefaultPrintLogger:
    def __init__(self):
        self.info = print
        self.error = print

class ProgressCallBackAbstract(metaclass=abc.ABCMeta):
    def __init__(self):
        self.stopping = False

    # must be implemented from user who uses this model
    def __call__(self, **kwargs) -> None:
        raise NotImplementedError

    # call when stopping
    def set_stop(self):
        self.stopping = True

    # use when model check whether stop or continue
    def is_stopping(self):
        return self.stopping
        
class GtGenModelAbstract(metaclass=abc.ABCMeta):
    def __init__(
                    self, 
                    devices:list = [0]
                    ):

        self.devices = devices
        self.model = None

    def has_model(self) -> bool:
        return self.model is not None

    def _assert_no_model(self):
        assert self.model is not None, "no loaded model"
        
    @abc.abstractmethod
    def get_model_config(self) -> dict:
        self.__assert_no_model()
        raise NotImplementedError

    @abc.abstractmethod
    # for MLOps
    # class names : ("vehicle", "pedestrian")
    def get_model_classes(self) -> tuple:
        self.__assert_no_model()
        raise NotImplementedError
        

class GtGenModelLearnAbstract(GtGenModelAbstract):
    def __init__(self, devices: list = [0]):
        super().__init__(devices)
        self.saved_model_path = None
        
    @abc.abstractmethod
    def create_dataloader(self,
                          phase:PhaseType,
                          **kwargs):
        raise NotImplementedError
        
    @abc.abstractmethod
    def train(self,
              train_dataloader,        
              valid_dataloader,
              pretrained_path,             
              resume:bool,
              progress_callback=None,
              save_path=None,
              **kwargs
              ):
        """
        best_loss=None
        best_model_path=None
        return best_loss, best_model_path
        """
        raise NotImplementedError
        
    def save_model(self, save_path) -> bool:
        self._assert_no_model()
        if self.saved_model_path:
            shutil.copyfile(self.saved_model_path, save_path)
            return True
        else:
            return False

class GtGenModelEvaluationAbstract(GtGenModelAbstract):
    def __init__(self, devices: list = [0]):
        super().__init__(devices)

    @abc.abstractmethod
    def load_model(self, load_path) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def evaluation(self,
                   progress_callback=None,
                   **kwargs) -> dict:
        raise NotImplementedError
    

class GtGenModelInferenceAbstract(GtGenModelAbstract):
    def __init__(self, devices: list = [0]):
        super().__init__(devices)

    @abc.abstractmethod
    def load_model(self, load_path) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def inference(self,
                  parent_path,
                  input_path_list: list,
                  progress_callback=None,
                  ) -> list:
        raise NotImplementedError