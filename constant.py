#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import scipy.constants 

au2ev = scipy.constants.physical_constants["Hartree energy in eV"][0]

cm2au = 1.0E2 * \
scipy.constants.physical_constants["inverse meter-hertz relationship"][0] / \
scipy.constants.physical_constants["hartree-hertz relationship"][0]
