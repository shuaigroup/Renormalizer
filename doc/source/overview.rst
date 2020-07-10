An overview of Renormalizer
*****************************

Renormalizer is a quantum dynamics package based on density matrix
renormalization group (DMRG) and its time dependent formulation written in
Python. All (TD-)DMRG algorithms are implemented within the framework of matrix
product states (MPS) and matrix product operators (MPO). The package aims to
provide a powerful and efficient platform for quantum dynamics simulation
and method development with (TD-)DMRG.  
The package is still under heavy development. Please refer to our recent papers
for the new developments.


How to cite
===========
If you use Renormalizer in your research, please cite

Bibtex entry::

    @article{ren2018time,
      title={Time-dependent density matrix renormalization group algorithms for nearly exact absorption and fluorescence spectra of molecular aggregates at both zero and finite temperature},
      author={Ren, Jiajun and Shuai, Zhigang and Kin-Lic Chan, Garnet},
      journal={Journal of chemical theory and computation},
      volume={14},
      number={10},
      pages={5027--5039},
      year={2018},
      publisher={ACS Publications}
    }

    @article{li2020numerical,
      title={Numerical assessment for accuracy and GPU acceleration of TD-DMRG time evolution schemes},
      author={Li, Weitang and Ren, Jiajun and Shuai, Zhigang},
      journal={The Journal of Chemical Physics},
      volume={152},
      number={2},
      pages={024127},
      year={2020},
      publisher={AIP Publishing LLC}
    }

If you also use the finite temperature dynamical DMRG code, please cite

Bibtex entry::

    @article{jiang2020finite,
      title={Finite Temperature Dynamical Density Matrix Renormalization Group for Spectroscopy in Frequency Domain},
      author={Jiang, Tong and Li, Weitang and Ren, Jiajun and Shuai, Zhigang},
      journal={The Journal of Physical Chemistry Letters},
      volume={11},
      number={10},
      pages={3761--3768},
      year={2020},
      publisher={ACS Publications}
    }


Features
===========

* Pure matrix product states (MPS) and matrix product operators (MPO) structure.

* Ground state calculation.
  
* Excited state with state-averaged DMRG algorithm.

* Wavefuntion and density matrix propagation.

* Real-time and imaginary-time propagation.

* Support GPU acceleration.

