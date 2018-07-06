# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>


class Electron(object):

    def __int__(self):
        return 1

    def __repr__(self):
        return 'e'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __ne__(self, other):
        return not self == other


electron = Electron()


class Phonon(object):

    def __int__(self):
        return 0

    def __repr__(self):
        return 'ph'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __ne__(self, other):
        return not self == other


phonon = Phonon()

