![logo](./doc/source/logo.png)

[![CircleCI](https://circleci.com/gh/shuaigroup/Renormalizer.svg?style=svg)](https://app.circleci.com/pipelines/github/shuaigroup/Renormalizer)
[![codecov](https://codecov.io/gh/shuaigroup/Renormalizer/branch/master/graph/badge.svg?token=T266FE7X9S)](https://codecov.io/gh/shuaigroup/Renormalizer)
[![pypi](https://img.shields.io/pypi/v/renormalizer.svg?logo=pypi)](https://pypi.org/project/renormalizer/)
[![doc](https://img.shields.io/badge/docs-link-green.svg)](https://shuaigroup.github.io/Renormalizer/)

Renormalizer is a Python based tensor network package with a special focus on electron-phonon quantum dynamics.

# Installation
```
pip install renormalizer
```

For users who are not familiar with python, 
please check out the step-by-step [installation guide](https://shuaigroup.github.io/Renormalizer/install.html).

# Features
- MPS/MPO based ground state search/excited state search/time evolution/dynamical properties
- TTNS/TTNO based ground state search/time evolution
- Custom Hamiltonian through automatic MPO/TTNO construction
- Finite temperature time evolution through imaginary time evolution or thermofield transformation
- Out-of-box molecular spectra/carrier mobility/spin relaxation dynamics/ab initio calculation.
- GPU acceleration via CuPy
- Quantum number conservation
- Purely Python based and eazy installation

# Documentation
Please check out the [official documentation](https://shuaigroup.github.io/Renormalizer/)
and the [examples](https://github.com/shuaigroup/Renormalizer/tree/master/example).

# Quickstart
```python
from renormalizer import Mps, Mpo, Op, Model, BasisHalfSpin

# a two-spin system. 0 and 1 denote the label of the spin
basis = [BasisHalfSpin(0), BasisHalfSpin(1)]
# the Hamiltonian has two terms, \sigma^+_0 \sigma^-_1 + \sigma^+_1 \sigma^-_0
ham_terms = Op("sigma_+ sigma_-", [0, 1]) + Op("sigma_+ sigma_-", [1, 0])
# build our model
model = Model(basis, ham_terms)
# the initial state: the 0th spin at the [0, 1] state
# the 1th spin will by default at the [1, 0] state
mps = Mps.hartree_product_state(model, condition={0:[0, 1]})
# the Hamiltonian
mpo = Mpo(model)
# the operator to calculation expectation
z_op = Mpo(model, Op("Z", 0))
# perform the time evolution
for i in range(10):
    mps = mps.evolve(mpo, 0.05)
    print(mps.expectation(z_op))
```


# Notice
Renormalizer relies on linear algebra libraries such as OpenBLAS or MKL for matrix operations. These libraries 
*by default* parallel over as many CPU cores as possible. 
However, we have found empirically that for calculations carried out in Renormalizer such parallelism has limited efficiency.
More specifically, the computational wall time hardly decrease if more than 4 cores are used 
(see this [paper](https://github.com/liwt31/publications/raw/master/2020numerical.pdf)).

> [!IMPORTANT]  
> We highly recommend limiting the number of parallel CPU cores to 4 for large scale calculations and 1 for small scale tests

To do this, set the environment variable before running Python
```bash
export RENO_NUM_THREADS=1
```
which sets all environment variables for underlying linear algebra libraries, such as `MKL_NUM_THREADS`.

> [!IMPORTANT]  
> After importing NumPy or Renormalizer, setting the environment variables will have no effect.

In fact, limiting the cores was once the default behavior of Renormalizer.
It is later changed in this [PR](https://github.com/shuaigroup/Renormalizer/pull/132) 
because Renormalizer is sometimes imported as a utility package.

# Useful links
- [The Shuai group](http://www.shuaigroup.net/)
- [tensornetwork.org](https://tensornetwork.org/)
- [A detailed review of our method](http://www.shuaigroup.net/images/article/pubs/2022/08_shuai_WIRES_Comput_Mol_Sci_2022_e1614.pdf)

# Papers published using Renormalizer

1. Qi Sun, Jiajun Ren, Qian Peng, and Zhigang Shuai*, 
[Heterofission Mechanism for Pure Organic Room Temperature Phosphorescence](http://www.shuaigroup.net/images/article/pubs/2023/09_shuai_Adv_Optical_Mater_2023_2301769.pdf). 
*Adv. Optical Mater.*, **2023**, 2301769.
2. Hengrui Yang, Weitang Li, Jiajun Ren, and Zhigang Shuai*, 
[Time-Dependent Density Matrix Renormalization Group Method for Quantum Transport with Phonon Coupling in Molecular Junction](http://www.shuaigroup.net/images/article/pubs/2023/05_shuai_JCTC_2023_19_6070.pdf). 
*J. Chem. Theory Comput.*, **2023**, *19*, 6070-6081.
3. Weitang Li*, Jonathan Allcock, Lixue Cheng, Shi-Xin Zhang, Yu-Qin Chen, Jonathan P. Mailoa, Zhigang Shuai, and Shengyu Zhang*, 
[TenCirChem: An Efficient Quantum Computational Chemistry Package for the NISQ Era](http://www.shuaigroup.net/images/article/pubs/2023/04_shuai_JCTC_2023_19_3966.pdf). 
*J. Chem. Theory Comput.*, **2023**, *19*, 3966−3981. 
4. Yuanheng Wang, Jiajun Ren* & Zhigang Shuai*, 
[Minimizing non-radiative decay in molecular aggregates through control of excitonic coupling](http://www.shuaigroup.net/images/article/pubs/2023/03_shuai_NatComm_2023_14_5056.pdf). 
*Nature Communications*, **2023**, *14*, 5056.
5. Weitang Li, Jiajun Ren, Sainan Huai, Tianqi Cai, Zhigang Shuai*, and Shengyu Zhang*, 
[Efficient quantum simulation of electron-phonon systems by variational basis state encoder](http://www.shuaigroup.net/images/article/pubs/2023/02_shuai_PhysRevResearch_2023_5_023046.pdf). 
*Phys. Rev. Research*, **2023**, *5*,  023046.
6. Tong Jiang, Jiajun Ren, and Zhigang Shuai*, 
[Unified Definition of Exciton Coherence Length for Exciton−Phonon Coupled Molecular Aggregates](http://www.shuaigroup.net/images/article/pubs/2023/01_shuai_JPCL_2023_14_4541.pdf). 
*J. Phys. Chem. Lett.*, **2023**, *14*, 4541−4547.
7. Yufei Ge, Weitang Li, Jiajun Ren, and Zhigang Shuai*, 
[Computational Method for Evaluating the Thermoelectric Power Factor for Organic Materials Modeled by the Holstein Model: A Time-Dependent Density Matrix Renormalization Group Formalism](http://www.shuaigroup.net/images/article/pubs/2022/18_shuai_JCTC_2022_18_6437.pdf). 
*J. Chem. Theory Comput.*, **2022**, *18*, 6437-6446.
8. Yuanheng Wang, Jiajun Ren, Weitang Li, and Zhigang Shuai*, 
[Hybrid Quantum-Classical Boson Sampling Algorithm for Molecular Vibrationally Resolved Electronic Spectroscopy with Duschinsky Rotation and Anharmonicity](http://www.shuaigroup.net/images/article/pubs/2022/11_shuai_JPCL_2022_13_6391.pdf). 
*J. Phys. Chem. Lett.*, **2022**, *13*, 6391−6399. 
9. Weitang Li, Jiajun Ren, Hengrui Yang and Zhigang Shuai*,
[On the fly swapping algorithm for ordering of degrees of freedom in density matrix renormalization group](http://www.shuaigroup.net/images/article/pubs/2022/10_shuai_JPhysCondensMatter_2022_34_254003.pdf). 
*J. Phys.: Condens. Matter.*, **2022**, *34*, 254003. 
10. Xing Gao*, Jiajun Ren, Alexander Eisfeld, and Zhigang Shuai, 
[Non-Markovian stochastic Schrödinger equation: Matrix-product-state approach to the hierarchy of pure states](http://www.shuaigroup.net/images/article/pubs/2022/09_shuai_PhysRevA_2022_105_L030202.pdf). 
*PHYSICAL REVIEW A*, **2022**, *105*, L030202. 
11. Jiajun Ren*, Weitang Li, Tong Jiang, Yuanheng Wang, Zhigang Shuai*, 
[Time-dependent density matrix renormalization group method for quantum dynamics in complex systems](http://www.shuaigroup.net/images/article/pubs/2022/08_shuai_WIRES_Comput_Mol_Sci_2022_e1614.pdf). 
*WIREs Comput Mol Sci.*, **2022**, e1614. 
12. Jia-jun Ren, Yuan-heng Wang, Wei-tang Li, Tong Jiang, Zhi-gang Shuai*, 
[Time-Dependent Density Matrix Renormalization Group Coupled with n-Mode Representation Potentials for the Excited State Radiationless Decay Rate: Formalism and Application to Azulene](http://www.shuaigroup.net/images/article/pubs/2021/22_shuai_CJCP_2021_34_565.pdf). 
*Chinese Journal of Chemical Physics*, **2021**, *34(5)*, 565-582. 
13. Tong Jiang, Jiajun Ren*, and Zhigang Shuai*, 
[Chebyshev Matrix Product States with Canonical Orthogonalization for Spectral Functions of Many-Body Systems](http://www.shuaigroup.net/images/article/pubs/2021/16_shuai_JPCL_2021_12_9344.pdf). 
*J. Phys. Chem. Lett.*, **2021**, *12*, 9344−9352. 
14. Weitang Li, Jiajun Ren, Zhigang Shuai*, 
[A general charge transport picture for organic semiconductors with nonlocal electron-phonon couplings](http://www.shuaigroup.net/images/article/pubs/2021/13_shuai_NC_2021_12_4260.pdf). 
*Nature Communications*, **2021**, *12*, 4260.
15. Yuanheng Wang, Jiajun Ren*, and Zhigang Shuai*, 
[Evaluating the anharmonicity contributions to the molecular excited state internal conversion rates with finite temperature TD-DMRG](http://www.shuaigroup.net/images/article/pubs/2021/09_shuai_JCP_2021_154_214109.pdf). 
*J. Chem. Phys.*, **2021**, *154*, 214109. 
16. Jiajun Ren*, Weitang Li, Tong Jiang, and Zhigang Shuai, 
[A general automatic method for optimal construction of matrix product operators using bipartite graph theory](http://www.shuaigroup.net/images/article/pubs/2020/12_shuai_JCP_2020_153_084118.pdf). 
*J. Chem. Phys.*, **2020**, *153*, 084118.
17. Weitang Li, Jiajun Ren, and Zhigang Shuai*, 
[Finite-Temperature TD-DMRG for the Carrier Mobility of Organic Semiconductors](http://www.shuaigroup.net/images/article/pubs/2020/08_shuai_JCPL_2020_11_p4930.pdf). 
*J. Phys. Chem. Lett.*, **2020**, *11*, 4930−4936.
18. Tong Jiang, Weitang Li, Jiajun Ren*, and Zhigang Shuai*, 
[Finite Temperature Dynamical Density Matrix Renormalization Group for Spectroscopy in Frequency Domain](http://www.shuaigroup.net/images/article/pubs/2020/03_shuai_JPCL_2020_11_p3761.pdf). 
*J. Phys. Chem. Lett.*, **2020**, *11*, 3761−3768. 
19. Weitang Li, Jiajun Ren*, and Zhigang Shuai, 
[Numerical assessment for accuracy and GPU acceleration of TD-DMRG time evolution schemes](http://www.shuaigroup.net/images/article/pubs/2020/01_shuai_JCP_2020_152_024127.pdf). 
*J. Chem. Phys.*, **2020**, *152*, 024127.
20. Jiajun Ren, Zhigang Shuai*, and Garnet Kin-Lic Chan*, 
[Time-Dependent Density Matrix Renormalization Group Algorithms for Nearly Exact Absorption and Fluorescence Spectra of Molecular Aggregates at Both Zero and Finite Temperature](http://www.shuaigroup.net/images/article/pubs/2018/14-shuai-2018-JCTC-14-p5027.pdf). 
*J. Chem. Theory Comput.*, **2018**, *14*, 5027-5039.