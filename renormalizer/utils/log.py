# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Wetiang Li <liwt31@163.com>

import logging
from logging import ERROR, WARN, INFO, DEBUG

import numpy as np

package_logger = logging.getLogger("renormalizer")
default_stream_handler = logging.StreamHandler()
default_formatter = logging.Formatter("%(asctime)s[%(levelname)s] %(message)s")


def init_log(level=logging.DEBUG):
    package_logger.setLevel(level)

    default_stream_handler.setLevel(logging.DEBUG)

    default_stream_handler.setFormatter(default_formatter)

    package_logger.addHandler(default_stream_handler)


def set_stream_level(level):
    default_stream_handler.setLevel(level)


def disable_stream_output():
    if default_stream_handler in package_logger.handlers:
        package_logger.removeHandler(default_stream_handler)


def register_file_output(file_path, mode="w", level=DEBUG):
    file_handler = logging.FileHandler(file_path, mode=mode)
    file_handler.setLevel(level)
    file_handler.setFormatter(default_formatter)
    file_handler.addFilter(logging.Filter("renormalizer"))
    package_logger.addHandler(file_handler)


NP_ERRCONFIG = {"divide": "raise", "over": "raise", "under": "ignore", "invalid": "raise"}

DEFAULT_NP_ERRCONFIG = np.seterr(**NP_ERRCONFIG)
