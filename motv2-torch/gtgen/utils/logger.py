#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import datetime
import logging
from logging import handlers as log_handlers

import traceback


class LoggerWrapper:
    def info(self):
        pass

    def error(self):
        pass


def setup_logger(logger_name,
                 log_prefix_name=None,
                 level=logging.INFO,
                 folder='.',
                 time_filename=False,
                 filename=None,
                 time_rotating=True,
                 backup_count=1,
                 max_bytes=1048576,  # 1 mega bytes
                 logging_=True,
                 console_=True):

    if not logging_:
        return get_stdout_logger()

    if not os.path.exists(folder):
        os.makedirs(folder)

    # 로거 파일은 사용자가 지정하는 고정적인 파일 이름 대신 date 와 time 으로 구성하는 것이 자동 로거를 구현할때 일반적이다.
    # 이전에 test.log 를 로거 파일 이름으로 설정한 이유는 단순히 테스트를 위한 것이다.
    # 아래에 filename 이 있을 경우와 없을 경우를 나누어서 로거의 파일 이름을 설정하도록 변경한다.
    if filename is None:
        if not log_prefix_name:
            filename = logger_name
        else:
            filename = log_prefix_name

    if time_filename:
        filename += '.' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename += '.log'

    log_path = os.path.join(folder, filename)

    if os.path.isfile(log_path):
        os.remove(log_path)
    
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(name)-10s | %(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # log 파일이 무한정 커지는 것을 방지하고, 서버 시작후 부터 일자별로 최대 backup_count(기본=0) 일까지 저장하도록 한다.
    if time_rotating:
        file_handler = log_handlers.TimedRotatingFileHandler(log_path, backupCount=backup_count, when='midnight')
    else:
        file_handler = log_handlers.RotatingFileHandler(log_path, mode='w',
                                                        backupCount=backup_count, maxBytes=max_bytes)

    file_handler.setFormatter(formatter)
    # file_handler.suffix = "%Y%m%d.bak"    # 이것 때문에 백업이 무한대가 됨.
    stream_handler = None
    if console_:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(file_handler)
    if console_ and stream_handler:
        log_setup.addHandler(stream_handler)
    return logging.getLogger(logger_name)


def write_exception_log(logger, e, msg=""):
    if not logger:
        logger = get_stdout_logger()

    if msg:
        msg += " : "
    msg += str(e) + "\n" + traceback.format_exc()
    logger.error(msg)


def get_stdout_logger(logger=None):
    if logger is None:
        logger = LoggerWrapper()
        logger.info = print
        logger.error = print
    return logger


def get_critical_logger(logger=None):
    if __debug__:
        if logger is None:
            return get_stdout_logger()
        return logger
    else:
        return get_stdout_logger()