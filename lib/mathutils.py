import numpy as N
#dtype=None
dtype=N.complex128

def zeros(shape):
    # allow us to set the default zero matrix type
    # (e.g. real or complex) by setting utils.dtype at beginning
    # of program
    return N.zeros(shape,dtype=dtype)


def Gram_Schmit(r, V):
    '''
    input:
        r is a vector 
        V is a list of vector 
    return:
        a vector orthogonal to the vector list r
    '''
    rnew = r.copy()
    for iv in V:
        overlap = rnew.dot(iv)
        rnew -= overlap * iv

    return rnew
