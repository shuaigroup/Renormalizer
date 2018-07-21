import numpy as np
import matplotlib.pyplot as plt


def tcorr(filename, dt, label):
    corr = np.load(filename)
    nsteps = len(corr)
    xplot = [i * dt for i in range(nsteps)]
    plt.plot(xplot, np.real(corr), label="real," + label + ",nsteps=" + str(nsteps))
    plt.plot(xplot, np.imag(corr), label="imag")


if __name__ == '__main__':
    tcorr("0.npy", 5, "0")
    tcorr("1.npy", 5, "1")
    # tcorr("22.npy",5,"00")
    # tcorr("32.npy",5,"11")
    # tcorr("2.npy",5,"0")
    # tcorr("3.npy",5,"1")

    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('time / au')
    # plt.xlim(0,100)
    # plt.legend()
    # plt.tick_params(labelleft='off')
    plt.show()
    # plt.savefig("emiTcorr.eps",bbox_extra_artists=(lgd,), bbox_inches='tight')
