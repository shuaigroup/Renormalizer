# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>


class MpAdaptor(object):

    def __init__(self, mp):
        self.mp = mp

    def __getattr__(self, item):
        if hasattr(self.mp, item):
            attr = getattr(self.mp, item)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    ret = attr(*args, **kwargs)
                    if isinstance(ret, self.mp.__class__):
                        ret = self.__class__(ret)
                    return ret
                return wrapper
            else:
                return attr
        else:
            raise AttributeError