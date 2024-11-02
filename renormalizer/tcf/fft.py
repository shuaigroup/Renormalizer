from renormalizer.utils.constant import *
import numpy as np
import scipy.integrate
import scipy.fftpack as fft
import logging
import scipy.signal
import scipy.interpolate

logger = logging.getLogger("renormalizer")

coeff_abs = 2*np.pi/3/speed_of_light
coeff_emi = 2/np.pi/3/speed_of_light**3

def ct2cw(fname, broaden_constant=None, broaden_func="gaussian",
        fft_type="backward", offset=None, interpolate=False, window=None,
        ic_rate=False, nsteps=None):
    """
    this is only for electronic emission/absorption spectrum calculation
    absorption: fft_type should be backward
    emission: fft_type should be forward, otherwise the w should be reversed
    manually since w_f - w_i is negetive.
    all units are atomic units
    """
    if type(fname) == str:
        res = np.load(fname)
        ct = res["ACF"]
        t = res["correlation function time"]
    elif type(fname) == tuple:
        ct, t = fname
    else:
        assert False
    
    if offset is not None:
        ct *= np.exp(1j*offset*t)
    
    if interpolate:
        f = scipy.interpolate.interp1d(t, ct, kind="cubic")
        #f = scipy.interpolate.CubicSpline(t, ct)
        t = np.linspace(t[0],t[-1],(len(t)-1)*10+1)
        ct = f(t)
    
    if nsteps is not None:
        ct = ct[:nsteps]
        t = t[:nsteps]

    if broaden_constant is not None:
        if broaden_func == "gaussian":
            ct *= np.exp(-t**2/2*broaden_constant**2)
        elif broaden_func == "lorentzian":
            ct *= np.exp(-t*broaden_constant)
            
    # extend to the negative axis
    nsteps = len(t)
    if np.allclose(t[0], 0):
        t2 = np.zeros(nsteps*2-1)
        t2[:nsteps] = -t[::-1]
        t2[nsteps:] = t[1:]

        ct2 = np.zeros(len(t2), dtype=np.complex128)
        ct2[len(t2)//2:] = ct
        ct2[:len(t2)//2] = ct[1:][::-1].conj()
    else:
        t2 = np.zeros(nsteps*2)
        t2[:nsteps] = -t[::-1]
        t2[nsteps:] = t

        ct2 = np.zeros(len(t2), dtype=np.complex128)
        ct2[len(t2)//2:] = ct
        ct2[:len(t2)//2] = ct[::-1].conj()

    if window is not None:
        ct2 *= window(len(t2))
    
    if fft_type == "backward":
        cw = fft.fftshift(fft.ifft(ct2))*(t2[1]-t2[0])*len(t2)   # +i\omega t backward
        w = fft.fftshift(fft.fftfreq(len(t2),t2[2]-t2[1]))*2*np.pi
        cw *= np.exp(1.0j*w*t2[0])
    elif fft_type == "forward":
        cw = fft.fftshift(fft.fft(ct2))*(t2[1]-t2[0])   # -i\omega t forward
        w = fft.fftshift(fft.fftfreq(len(t2),t2[2]-t2[1]))*2*np.pi
        cw *= np.exp(-1.0j*w*t2[0])
        
    # calculate internal conversion rate
    if ic_rate:
        assert fft_type == "backward"
        cumu_int = scipy.integrate.cumulative_trapezoid(ct2.real, x=t2, initial=0) / au2fs / 1e-15
        logger.info(f"ic rate with cumulative_trapezoid (last three): {cumu_int[-3:]}")
        simpson_int = scipy.integrate.simpson(ct2.real, x=t2) / au2fs / 1e-15
        logger.info(f"ic rate with simpson_integration: {simpson_int}")
        # the ic rate can also be read from the cw with w=0, w can be understood
        # as the shift of the adiabatic excitation energy, w>0 means shift up
        # (with backward ffttype)
        return ct2, t2, cw/au2fs/ 1e-15, w, cumu_int, simpson_int
    else:
        return ct2, t2, cw, w

def emi_rate(cw, w, coeff=True):
    """
    emission rate unit: s^-1
    """
    index = np.argmin(np.abs(w))
    logger.debug(f"the zero value index: {index}, {w[index]}")
    if coeff:
        y = (cw.real*w**3*coeff_emi)[index:]
    else:
        y = cw.real[index:]

    x = w[index:]
    cumu_int = scipy.integrate.cumulative_trapezoid(y, x=x, initial=0) / au2fs / 1e-15
    logger.info(f"emission rate with cumulative_trapezoid (last three): {cumu_int[-3:]}")
    simpson_int = scipy.integrate.simpson(y, x=x) / au2fs / 1e-15
    logger.info(f"emission rate with simpson_integration: {simpson_int}")
    return cumu_int, simpson_int

