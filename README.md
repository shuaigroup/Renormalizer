![logo](./doc/source/logo.png)

[![CircleCI](https://circleci.com/gh/shuaigroup/Renormalizer.svg?style=svg)](https://app.circleci.com/pipelines/github/shuaigroup/Renormalizer)
[![codecov](https://codecov.io/gh/shuaigroup/Renormalizer/branch/master/graph/badge.svg?token=T266FE7X9S)](https://codecov.io/gh/shuaigroup/Renormalizer)

Renormalizer is a python package based on tensor network states for electron-phonon quantum dynamics.

## Installation
```
pip install renormalizer
```

For users who are not familiar with python, step-by-step installation guide can be found in the [document](https://github.com/shuaigroup/Renormalizer/wiki/Installation-guide).

## Documentation
Primitive documentation could be found [here](https://shuaigroup.github.io/Renormalizer/).

## Notice
Renormalizer relies on linear algebra libraries such as OpenBLAS or MKL for matrix operations. These libraries 
by default parallel over as many CPU cores as possible. 
However, we have found empirically that the calculations carried out in Renormalizer has **very poor** parallelism efficiency over 4 cores 
(see this [paper](https://github.com/liwt31/publications/raw/master/2020numerical.pdf)).
Thus, we **highly** recommend limiting the number of parallel CPU cores to 4 for large scale calculations and 1 for small scale tests.
To do this, set the environment variable before running Python
```bash
export RENO_NUM_THREADS=1
```
which sets all environment variables for underlying linear algebra libraries, such as `MKL_NUM_THREADS`.

>  After importing NumPy or Renormalizer, setting the environment variables will have no effect.

In fact, limiting the cores was once the default behavior of Renormalizer.
It is later changed in this [PR](https://github.com/shuaigroup/Renormalizer/pull/132) 
because Renormalizer is sometimes imported as a utility package.
