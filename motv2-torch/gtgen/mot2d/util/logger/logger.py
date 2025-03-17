import os
import json
import logging
from collections import OrderedDict
from .base import LoggerBase
from abc import *

logger_initialized = {}
LOG_MAX_SIZE = 1024 * 1024 * 10
LOG_FILE_CNT = 10

class LogBuffer:
    def __init__(self):
        self.history    =   list()
        self.output     =   OrderedDict()

    def clear(self):
        self.history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()

    def update(self, vars):
        assert isinstance(vars, dict)

        #self.clear_output()
        history_dict    =   dict()
        for key, var in vars.items():
            history_dict[key]   =   var
            self.output[key]    =   var

        self.history.append(history_dict)

    def get_output_dict(self, log_dict):
        return dict(log_dict, **self.output)

class Logger(LoggerBase):
    def __init__(
                self, 
                name            =   'log', 
                log_file        =   None,
                log_json_file   =   None,
                json_save       =   False, 
                log_level       =   logging.INFO, 
                file_mode       =   'w',
                interval        =   10,
                log_max_size    =   1024 * 1024 * 10,
                log_file_cnt    =   10,
                format          =   '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ):
    
        if log_file == None:
            print("File path is wrong.")
            raise

        self.logger = self._get_logger(
                        name = name, 
                        log_file = log_file, 
                        log_level = log_level, 
                        file_mode = file_mode,
                        format = format,
                        log_max_size = log_max_size,
                        log_file_cnt = log_file_cnt 
                        )

        self.log_json_file  =   log_json_file
        self.json_save      =   json_save
        self.interval       =   interval
        self.format         =   format
        self.log_buffer     =   LogBuffer()

        super(Logger, self).__init__()

    def update_buffer(self, vars):     

        self.log_buffer.update(vars)
        log_dict    =   OrderedDict()

        if len(self.log_buffer.history) % self.interval == 0:
            output              =   self.log_buffer.get_output_dict(log_dict)
            history             =   self.log_buffer.history
            json_string_output  =   json.dumps(output)
            self.print_log(json_string_output)

            if self.json_save : 
                try :
                    with open(self.log_json_file, 'a+') as json_file:
                        for hs in history:
                            hs_string   =   json.dumps(hs)
                            json_file.write(hs_string)
                            json_file.write('\n')
                except IOError:
                    msg     =   ("Unable to create file on disk.")
                    self.print_log(msg, level = logging.ERROR)
                finally :
                    self.log_buffer.clear()


    def _get_logger(self,
                    name, 
                    log_file=None, 
                    log_level=logging.INFO, 
                    file_mode='w', 
                    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    log_max_size = 1024 * 1024 * 10,
                    log_file_cnt = 10  
                    ):
        """Initialize and get a logger by name.
        If the logger has not been initialized, this method will initialize the
        logger by adding one or two handlers, otherwise the initialized logger will
        be directly returned. During initialization, a StreamHandler will always be
        added. If `log_file` is specified and the process rank is 0, a FileHandler
        will also be added.
        Args:
            name (str): Logger name.
            log_file (str | None): The log filename. If specified, a FileHandler
                will be added to the logger.
            log_level (int): The logger level. Note that only the process of
                rank 0 is affected, and other processes will set the level to
                "Error" thus be silent most of the time.
            file_mode (str): The file mode used in opening log file.
                Defaults to 'w'.
        Returns:
            logging.Logger: The expected logger.
        """
        logging.basicConfig(level = log_level,
                    format = format,
                    datefmt = '%m-%d %H:%M',
                    filename = log_file,
                    filemode = file_mode)

        logger = logging.getLogger(name)
        if name in logger_initialized:
            return logger
        # handle hierarchical names
        # e.g., logger "a" is initialized, then logger "a.b" will skip the
        # initialization since it is a child of "a".
        for logger_name in logger_initialized:
            if name.startswith(logger_name):
                return logger

        # handle duplicate logs to the console
        # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
        # to the root logger. As logger.propagate is True by default, this root
        # level handler causes logging messages from rank>0 processes to
        # unexpectedly show up on the console, creating much unwanted clutter.
        # To fix this issue, we set the root logger's StreamHandler, if any, to log
        # at the ERROR level.
        for handler in logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]

        # only rank 0 will add a FileHandler
        if log_file is not None:
            # Here, the default behaviour of the official logger is 'a'. Thus, we
            # provide an interface to change the file mode to the default
            # behaviour.
            file_handler = logging.FileHandler(log_file, file_mode)
            handlers.append(file_handler)

        rotate_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes = log_max_size, backupCount =  log_file_cnt)
        handlers.append(rotate_handler)

        formatter = logging.Formatter(format)

        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)

        logger.setLevel(log_level)
        logger_initialized[name] = True

        return logger


    def print_log(self, msg, level=logging.INFO):
        """Print a log message.
        Args:
            msg (str): The message to be logged.
            logger (logging.Logger | str | None): The logger to be used.
                Some special loggers are:
                - "silent": no message will be printed.
                - other str: the logger obtained with `get_root_logger(logger)`.
                - None: The `print()` method will be used to print log messages.
            level (int): Logging level. Only available when `logger` is a Logger
                object or "root".
        """
        if self.logger is None:
            print(msg)
        elif isinstance(self.logger, logging.Logger):
            self.logger.log(level, msg)
        elif self.logger == 'silent':
            pass
        elif isinstance(self.logger, str):
            _logger = self._get_logger(self.logger)
            _logger.log(level, msg)
        else:
            raise TypeError(
                'logger should be either a logging.Logger object, str, '
                f'"silent" or None, but got {type(self.logger)}')