This is a electron-phonon hamiltoninan MPS

[![Build Status](https://travis-ci.org/jjren/ephMPS.svg?branch=wtli-develop)](https://travis-ci.org/jjren/ephMPS)

## Installation
Installation guide can be found in the [project wiki](https://github.com/jjren/ephMPS/wiki/Installation-Guide).

## Notice
* ephMPS is still under heavy development. Drastic API changes come without any prior deprecation.
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

## thoughts:
* should separate evolution to a different file? no. tdvp touches the core of the data structure.
* lazy evaluation when mpo is applied on mps to save memory? Not useful. After `contract` canonicalise is performed,
lazy evaluation has no diff with eager evaluation.
* benchmark framework (could be useful in detecting bugs)？ No. Need to much maintenance. 
Currently use `pytest --durations=0` should work well.
* automatic switch from p&c to tdvp (only when p&c becomes slower? 
or when highest bond order hits expectated bond order?). Fancy but not useful. The user should have this kind of
control (by subclassing, etc.)

## todo
* Choose a baseline for core mps test. Better with analytical result.
* scheduler for CPU and GPU memory (CPU memory for small matrices and for doing SVD)
* TDVP uses scipy.linalg.qr. Update to Csvd. Better performance and preserves qn info. (`ps` has been updated to qn version.)
* investigate GPU usage fluctuation pattern and possibly optimize.
* license of libs
* include the tdh part with backend
