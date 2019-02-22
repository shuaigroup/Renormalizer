This is a electron-phonon hamiltonina MPS

## Notice
* use `-O` flag to enable optimizations (disable asserts) for Python can typically speed
things up for 50% (time cost drops to 66%).


## common mistakes during development

* Forget to convert from float datatype to complext datatype. Usually NumPy will issue a warning.

## todo
* benchmark framework (could be useful in detecting bugs)
* switch from p&c to tdvp more cleverly (only when p&c becomes slower? 
or when highest bond order hits expectated bond order?)
* refactor utils/tdmps. make economic mode the default. user should input what they
what to be calculated.