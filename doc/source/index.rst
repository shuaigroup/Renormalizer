.. image:: logo.png

Welcome to Renormalizer's documentation!
========================================
Renormalizer is a Python based tensor network package with a special focus on electron-phonon quantum dynamics.
It is originally developed based on matrix product states (MPS) and matrix product operators (MPO)
and thus the algorithms are called (TD-)DMRG.
In 2024 tree tensor network states (TTNS) and tree tensor network operators (TTNO) utilities are implemented.
Renormalizer can routinely perform quantum dynamics calculation involving hundreds to thousands of degrees of freedom
in a numerically exact manner, sometimes using only a single CPU core.

Renormalizer is developed by `Prof. Zhigang Shuai's group <http://www.shuaigroup.net/index.php>`_.
Its source code is hosted on `GitHub <https://github.com/shuaigroup/Renormalizer>`_.


Features
===========

* MPS/MPO based ground state search/excited state search/time evolution/dynamical properties
* TTNS/TTNO based ground state search/time evolution
* Custom Hamiltonian through automatic MPO/TTNO construction
* Finite temperature time evolution through imaginary time evolution or thermofield transformation
* Out-of-box molecular spectra/carrier mobility/spin relaxation dynamics/ab initio calculation
* GPU acceleration via CuPy
* Quantum number conservation
* Purely Python based and easy installation


Contents
--------

.. toctree::
   :maxdepth: 2
   :numbered:

   install.md
   tutorial.rst
   applications.rst
   api.rst
   faq.rst
   develop.md
   cite.rst




.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
