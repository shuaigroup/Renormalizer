import numpy as np
from ephMPS import constant

def select_mode(fname="dushin_cart.dus"):
    f = open(fname, "r")
    data = []
    for line in f:
        data.append([float(line.split()[3]),float(line.split()[4])])

    f.close()
    data = np.array(data)
    S = (np.sqrt(data[:,0]*constant.cm2au/2.)*data[:,1])**2
    gw = np.sqrt(data[:,0]*constant.cm2au/2.)*data[:,1]* data[:,0] *constant.cm2au
    idx = np.argsort(np.absolute(gw))
    print idx
    return data[idx[::-1],:]
    
