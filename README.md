This is a electron-phonon hamiltoninan MPS

[![Build Status](https://travis-ci.org/jjren/ephMPS.svg?branch=wtli-develop)](https://travis-ci.org/jjren/ephMPS)

## Notice
* use `-O` flag to enable optimizations (disable asserts) for Python can typically speed
things up for 50% (time cost drops to 66%).
* use float32/complex64 as backend (rather than float64/complex128) can speed things up for 50%. Although all test cases can pass
in such configuration, care should be taken because the precision is significantly lower.


## common mistakes during development

* Forget to convert from float datatype to complext datatype. Usually NumPy will issue a warning.

## why not use `opt_einsum`
* Overhead of finding path prohibits usage at critical points. Yes paths can be saved
but why not just hard code the contraction?
* Without prior knowledge on the contractions to be done, `shared_intermediates` must cost
a lot of memory.

## todo
* scheduler for CPU and GPU memory (CPU memory for small matrices and for doing SVD)
* TDVP uses scipy.linalg.qr. Update to Csvd. Better performance and preserves qn info. (`ps` has been updated to qn version.)
* investigate GPU usage fluctuation pattern and possibly optimize.
* benchmark framework (could be useful in detecting bugs)
* switch from p&c to tdvp more cleverly (only when p&c becomes slower? 
or when highest bond order hits expectated bond order?)
* separate evolution to a different file
* refactor utils/tdmps. make economic mode the default. user should input what they
what to be calculated. Need to ask jjr what is the best prctice. And start from chk file?
* license of libs
* different backend for original mctdh_tdvp is buggy
* include the tdh part with backend
* lazy evaluation when mpo is applied on mps to save memory