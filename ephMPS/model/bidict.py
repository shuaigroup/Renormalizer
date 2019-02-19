# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, print_function, unicode_literals


class bidict(dict):
    """
    bi-dictionary class, doule-way hash table
    """

    def __init__(self, *args, **kwargs):
        self.inverse = {}
        super(bidict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super(bidict, self).__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key):
        del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)
