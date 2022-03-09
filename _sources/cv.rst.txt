Frequency domain DMRG
*************************************

cv --- correction vector
=========================

Use correction vector to calculate spectral function (currently only auto-correlalation funtion,
general correlation function will be implemented later), here the spectral function is:

.. math:: I_{\mu\mu}=-\frac{1}{\pi}\rm{Im}\langle\Psi_0|\hat{\mu}^\dagger\frac{1}{\omega-\hat{H}+\omega_0)+i\eta}\hat{\mu}|\Psi_0\rangle.

at zero temperature, where :math:`\hat{\mu}` is the dipole operator here to calculate the absorption/emission spectrum,
properties such as optical conductivity is with current operator, also will be implemented soon. :math:`\eta` is a Lorentzian broaden width.

.. math:: I_{\mu\mu}=-\frac{1}{\pi}\rm{Im}\rm{Tr}\{\hat{\mu}\frac{1}{\omega-\hat{L}+i\eta}\hat{\mu}\hat{\rho}_{\beta}\}.

which is actually calculated in a symmetrized form:

.. math:: I_{\mu\mu}=-\frac{1}{\pi}\rm{Im}\rm{Tr}\{\hat{\rho}_{\beta}^{1/2}\hat{\mu}\frac{1}{\omega-\hat{L}+i\eta}\hat{\mu}\hat{\rho}_{\beta}^{1/2}\}.

at finite temperature, where :math:`\hat{\rho}_{\beta}^{1/2}` represents :math:`e^{-\beta H/2}`, which is obtianed from imaginary time evolution,
:math:`\hat{L}` is a Liouville operator.

.. automodule:: renormalizer.cv.zerot
   :members:


.. automodule:: renormalizer.cv.finitet
   :members:

