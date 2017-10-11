import numpy as np
from obj import *
import scipy.constants 
from constant import * 
from MPSsolver import *
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from ephMPS.exact_solver import spectra_normalize


def spectrum(autocorr,label):
    nsteps = autocorr.shape[0]
    print "nsteps=", nsteps
    
    nsteps = nsteps-omit
    autocorr = autocorr[0:nsteps]
    
    xplot = np.array([i*dt for i in range(nsteps)])
    autocorr = autocorr * np.exp(-(xplot/tau)**2)

    #plt.plot(xplot, np.real(autocorr))
    #plt.plot(xplot, np.imag(autocorr))
    #plt.figure()
    
    yf = fft.fft(autocorr)
    yplot = fft.fftshift(yf)
    xf = fft.fftfreq(nsteps,dt)
    
    # in FFT the frequency unit is cycle/s, but in QM we use radian/s,
    # hbar omega = h nu   omega = 2pi nu   
    xplot = fft.fftshift(xf) * 2.0 * np.pi
    
    if mode == "abs":
        xplot = -(xplot-e0)*convert
        yplot = yplot * xplot
    elif mode == "emi":
        xplot = (xplot+e0)*convert
        for i, ixplot in enumerate(xplot):
            if ixplot > left and ixplot < right:
                yplot[i] *= ixplot**3

    yplot = spectra_normalize(yplot)
    plt.plot(xplot, yplot, label=label)

tau = 50. * fs2au
autocorr = np.load("0Temi.npy")
dt = 5.0
omit = 0
left = 13000
right = 19000
plt.xlim(13000,19000)
#plt.ylim(0,0.15)
e0 = 2.08695947118/au2ev
convert = 1.0/cm2au
mode = "emi"
spectrum(autocorr,"0K")

autocorr = np.load("TTemi.npy")
dt = 1.0
spectrum(autocorr,"298K")

plt.xlabel("cm^-1")
plt.ylabel("Intensity(normalized)")

plt.legend()
plt.show()

#plt.savefig("2mol_0Temi.eps")

