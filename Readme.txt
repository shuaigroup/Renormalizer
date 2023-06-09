
A Doc For "Renormalizer"

Renormalizer is a python package based on tensor network states for electron-phonon quantum dynamics.
An online wiki for our package is available (https://shuaigroup.github.io/Renormalizer/overview.html).
Here is a "Readme.txt" doc specially designed for readers of the article "Minimizing non-radiative decay in molecular aggregates through optimal control of excitonic coupling"
This doc includes three sections (1.Installation. 2.Demos & Reproduction instructions  3.Further instructions)





1. Installation


An open source repository for our "Renormalizer "code is available: https://github.com/shuaigroup/Renormalizer.
We recommend the user to follow the online installation guide (https://shuaigroup.github.io/Renormalizer/install.html) to build the environment for our python3 code. While  "Renormalizer" is still under heavy development. Drastic API changes come without any prior deprecation. Currently it is necessary to use the source code provided here in "/Source Code/renormalizer" directory to overwrite the files with the same names in the github version source code (https://github.com/shuaigroup/Renormalizer) to get the demos given below running properly.

Or the user can directly copy the source code provided here in the "/Source Code/renormalizer" directory to your working directory, which is the most suitable version for calculations in this work. Then the user can follow the online installation guide to build the python environment required.

A python3 environment with required packages is required to run our code, we suggest the user use "Anaconda" (https://www.anaconda.com/) to build this environment.
For more detailed information related to the installation, including system requirements, installation guide and documentations, an online wiki is available
(https://shuaigroup.github.io/Renormalizer/install.html).


Here, we brief introduce some key information. It is highly recommended for the users to refer to the online wiki (https://shuaigroup.github.io/Renormalizer/install.html) when meeting difficulty during the installation.

## Installation

Step-by-step installation (For the users having already downloaded the soruce code or copied it from the "/Source Code" directory provided.)

A step-by-step installation guide can be found in the document(https://github.com/shuaigroup/Renormalizer/wiki/Installation-guide) or (https://shuaigroup.github.io/Renormalizer/install.html). The website give instructions on how to build a python environment with required packages to run our code. An off-line "Installation guide.txt" is also provided here in the directory.

## Required packages
numpy==1.22.*
scipy==1.8.*
pytest==6.2.*
h5py==3.1.*
PyYAML==5.4.*
recommonmark==0.7.*
furo==2022.9.*
nbsphinx==0.8.*
opt_einsum==3.3.*
qutip==4.6.*

## Documentation
Primitive documentation could be found here(https://shuaigroup.github.io/Renormalizer/).

## Installation time
Typical installation time on a "normal" desktop computer should be several minutes.





2. Demos & Reproduction instructions


Note: Currently it is necessary to use the source code provided here in "/Source Code/renormalizer" directory to overwrite the files with the same names if the user get the source code from the github (https://github.com/shuaigroup/Renormalizer) to get the Demos given below running properly.


Here we provide four demos showing how to use our "Renormalizer" to caculate the studied properties in this work and how to process them to reproduce the FIG.4a, FIG.5a FIG.5b and FIG.8b in the article.
For each calculation job of "Renormalizer", we provide the input file for "Renormalizer" (main.py) and expected output files from "Renormalizer" under the "data" directory (/Demo/data).  
Then we use a jupyter notebook file (XXX.ipynb) to process the data and plot each figure. To run a calculation and reproduce the data, the user should add the pre-installed "Renormalizer" to the system path or copy the "Renormalizer" source code to the the same directory as the input file "main.py". We give an example for the result of such copy job in the directory (/Demo/data/Dimer_zt/dimer_j_0lambda-1p1a). 
The user should be able to direclty run the job properly with:

python main.py

in this directory once the python environment required is properly built. (An online wiki: https://github.com/shuaigroup/Renormalizer/wiki/Installation-guide is prepared to help the user to build the python environment required, offline part only for installation is also provided "Installation guide.txt". ) Then "test.log", "test.npz" and "agg_s1_e_ph_correlation.npy" are expected to output. 


In most of the time, we use "Renormalizer" to calculate the time-correlation functions (TCF). Detailed defination of TCF and their usage in this work can be seen in the Section Method in the main article. 
For the typical run time benchmark, our benchmark platform is one CPU core on Intel Xeon® Gold 5115 CPU at 2.40 GHz with NVIDIA® Tesla® V100-PCIE-32 GB for CPU-GPU heterogeneous calculations. Linux system is used.
For the two-mode model aggregate systems, the typical run time should be below 10 minutes for a TCF job in a "Dimer" system (/Demo/data/Dimer_zt,/Demo/data/Dimer_rt), 
 below 1 hour for a zero temperature TCF job in a "1D chain" system (/Demo/data/1d_zt), around 10 hours for a room temeprature TCF job in a "1D chain" system (/Demo/data/1d_rt),
around 4 hours for a zero temperature TCF job in a "2D square lattice" system (/Demo/data/2d_zt) and around 1 day (depending on the bond dimensions M of matrix product states used) for a room temeprature TCF job in a "2D square lattice" system (/Demo/data/2d_rt).
For the azulene dimer systems, the typical run time should be around 3 hours for a zero temperature TCF job (/Demo/data/azulene).


2.1. FIG.5a: To calculate and visualize the aggregate non-radiative decay rate (k_nr)
Using the "Renormalizer", the user can run the input file "main.py" to calculate the time-correlation functions (TCF) for each system studied (dimer,1d-chain and 2d-square lattice with different excitonic coupling strength and at zero temperature or room temperature). Corresponding TCF is output in "test.npz" file in default.  Related input/output files used are under the "data" directory.
"FIG.5a.ipynb" show the process to perform time integation on TCF to obtatin the non-radiative decay rate k_nr and plot them as a figure. Detailed derivations for treatments done here is available in the Method section in the main article. 


2.2  FIG.5b: To calculate and visualize the aggregate reorganization energy (\widetilde{\lambda}) and the energy gap narrowing (\Delta E)
Note:
To calculate the aggregate reorganization energy, we need the vibrational distorsion field  (VDF,D(r)). The definations of the aggregate reorganization energy and the energy gap narrowing (\Delta E) and details for using VDF to calculat the aggregate reorganization energy is available in the section Results in the main article.

Based on the calculations already done in 2.1 (/Demo/data/Dimer_zt, /Demo/data/1d_zt, /Demo/data/2d_zt ), VDF  and energy gap narrowing is in fact also calculated when calculating the TCF."FIG.5b.ipynb" show how to collect the data to calcualte the energy gap narrowing (\Delta E) from "test.log" output file and the data to calculate the aggregate reorganization energy (\widetilde{\lambda}) from the "agg_s1_e_ph_correlation.npy" output file and visulize them. 


2.3. FIG.8b: To calculate and visualize the distribution of vibrational distorsion field (VDF,D(r))

Based on the calculations (/data/2d_zt, /data/2d_rt) already done in 2.1, "FIG.8b.ipynb" show the process to collect the VDF from the "agg_s1_e_ph_correlation.npy" file and visulize them.


2.4. FIG.4a: To calculate and visualize the non-radiative decay spectrum (k_nr vs E_ad) of azulene dimers.
As the last example, we demonstrate here how to caculate and visualize the non-radiative decay spectrum of the real molecule, the azulene dimer with the "Renormalizer".

Using "Renormalizer", we run the corresponding input file "main.py" to calculate the time-correlation functions (TCF) for azulene dimers.(/Demo/data/azulene)
The ab-intio physical parameters of azulene used here is packed in the "/par" directory and will be read by "main.py" in the same directory in default.
(An example with the source code also copied to the directory: /Demo/data/azulene/dimer-0j)
"FIG.4a.ipynb" show the process to perform time integation on TCF to obtatin the non-radiative decay rate (k_nr) for different E_ad values given. With series of E_ad values and their correpsonding k_nr, we can plot the non-radiative decay spectrum as shown in the FIG.4a.





3.Further instructions.
Other figures in the article can be obtained through the similar ways demonstrated above. So we choose not to introduce them one-by-one here for simplicity. Further reproduction instructions, the data and the code that support the findings of this study are available from the corresponding author upon reasonable request.(zgshuai@tsinghua.edu.cn)


