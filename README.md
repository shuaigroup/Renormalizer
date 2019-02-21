This is a electron-phonon hamiltonina MPS


## common mistakes during development

* Forget to convert from float datatype to complext datatype. Usually NumPy will issue a warning.

## todo
* benchmark framework (could be useful in detecting bugs)
* switch from p&c to tdvp more cleverly (only when p&c becomes slower? 
or when highest bond order hits expectated bond order?)