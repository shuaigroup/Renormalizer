# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Wetiang Li <liwt31@163.com>

import logging

logger = logging.root
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s[%(levelname)s] %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)