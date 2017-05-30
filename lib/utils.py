import numpy as N
#dtype=None
dtype=N.complex128

def zeros(shape):
    # allow us to set the default zero matrix type
    # (e.g. real or complex) by setting utils.dtype at beginning
    # of program
    return N.zeros(shape,dtype=dtype)
