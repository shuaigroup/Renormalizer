# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Wetiang Li <liwt31@163.com>

import logging
from logging import ERROR, WARN, INFO, DEBUG

root_logger = logging.root
default_stream_handler = logging.StreamHandler()
default_formatter = logging.Formatter('%(asctime)s[%(levelname)s] %(message)s')


def init_log(level=logging.DEBUG):
    root_logger.setLevel(level)

    default_stream_handler.setLevel(logging.DEBUG)

    default_stream_handler.setFormatter(default_formatter)

    root_logger.addHandler(default_stream_handler)


def set_stream_level(level):
    default_stream_handler.setLevel(level)


def disable_stream_output():
    if default_stream_handler in root_logger.handlers:
        root_logger.removeHandler(default_stream_handler)


def register_file_output(file_path, mode='w', level=DEBUG):
    file_handler = logging.FileHandler(file_path, mode=mode)
    file_handler.setLevel(level)
    file_handler.setFormatter(default_formatter)
    root_logger.addHandler(file_handler)
