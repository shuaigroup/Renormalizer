# Installation guide

## Quick installation
If you are familiar with Python and not concerned with the development of the package, install `renormalizer` via `pip`
```bash
pip install renormalizer
```

Optionally, install `primme` for high-performance static DMRG
```
pip install primme
```
If any error occurs during the installation, see the [primme official document](https://github.com/primme/primme) for details.

Optionally, install `CuPy` for GPU backend. Please see the [CuPy official document](https://docs.cupy.dev/en/stable/install.html) for details.
`renormalizer` auto-detects whether `CuPy` is installed during its import.
If `CuPy` is installed, then `renormalzier` will automatically use GPU to accelerate tensor contractions.

If you are not familiar with python or wish to setup a local development environment, continue reading.

## Install from scratch

This is a step by step guide on how to prepare an environment for Renormalizer from scratch. 
Although "install" is in the name of this section, we actually won't "install" Renormalizer as a standalone package that can be imported anywhere. What we achieve in this section is to put source files of Renormalizer into your computer and prepare a Python environment that can run it.

Renormalizer runs on both CPU and GPU. It is recommended to install CPU-only version first, then GPU support (if possible).

The CPU-only version runs on Windows/MacOSX/Linux, whereas the GPU support is only tested on Linux. 

As operations on MacOSX are similar to those on Linux, from now on we'll only talk about Windows and Linux.

### Python environment
There are various ways to prepare a Python environment for Renormalizer. Here we recommend using Anaconda (both 2 or 3 are ok, we recommend Anaconda 3) as a package and environment manager.
#### Why you need Anaconda when you already have Python
Anaconda manages *environments*. If you use Python frequently enough, one day you'll face problems similar to this: project A runs on Python 3.5 but not Python 3.6, project B runs only on Python 3.6, how could I develop project A and project B on the same machine? Anaconda can help you solve these problems with ease.

Another common issue about Python is that its environment screws up quite frequently. You may accidentally install multiple versions of the same package or delete some critical packages and then you'll get a strange error importing the package you want. Find out what went wrong can take hours and you just wish you can reinstall Python. Anaconda can help you reinstall Python clean and easy with 2-3 commands.

Anaconda has lots of other cool features, you'll learn more while you're using it.

If you find the installation of Anaconda annoying and you already have Python 3.8, in theory, you can skip the installation and use your Python directly. But if something went wrong and your Python couldn't work anymore, you'll probably feel that using Anaconda is a good idea.

#### Install Anaconda
You can skip this step if you already have access to Anaconda (which means you can use `conda` command in your command line). For Linux users, don't worry if you don't have the root privilege -- installing Anaconda doesn't require root.

> A reminder to *dirac* (The cluster in Shuai group) users: both Anaconda2 and Anaconda3 have been installed on the cluster. Try `module load anaconda/3`.

For users in Tsinghua, Anaconda can be downloaded from *[tuna](https://tuna.tsinghua.edu.cn/)*. You can simply think of *tuna* as a cloud disk inside Tsinghua.

There are lots of links in the [download page](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/). You need to choose a correct version for your operating system and preferably a relatively newer version.

If access to *tuna* is not feasible, please download Anaconda [here](https://www.anaconda.com/).

For Linux users, use `bash` to run the installation script.

After the installation, test the installation using `conda` command. Anaconda has a graphic user interface for MacOSX and Windows, however we recommend using a command-line interface. If your command line can't find `conda`, please try to locate where `conda` is installed and add `conda` (`conda.exe` for Windows) to your system environment `PATH`, or you can use `conda prompt` if it is installed.

#### Create an environment
If Anaconda has been installed you should now be able to use Python in the command line. This Python is shipped with Anaconda and we recommend **not** using it for any particular Python project. The best practice is to create an environment for every project so that they don't mess up. 

Users inside Tsinghua may use *tuna* to speed up installation. See [this page](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) for the official document. In short, use the following command:
``` 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/    
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/    
conda config --set show_channel_urls yes    
``` 
to let Anaconda know it can download resources from *tuna*.

In this step, we're going to create a Python environment for Renormalizer using Anaconda. Two things must be taken into consideration while creating a Python environment:
* Python version. **Python version higher than Python 3.5 is required for Renormalizer**. We recommend using Python 3.8 or higher. 
    > Renormalizer uses *type annotation* like `a: int = 3` or `b: float = 3.0` which helps to make code clearer but requires Python 3.6 or higher. For more info about *type annotation*, see the [typing module](https://docs.python.org/3/library/typing.html) or [PEP484](https://www.python.org/dev/peps/pep-0484/).
* Name of the environment. Choose anything you like. **In this guide we are going to use `Reno` as it is the nickname of the project**.

With Python version and name of the environment determined, create the environment using the following command (Note the environment name `Reno`)
```
conda create -n Reno python=3.8 -y
```
This might take 1 to 20 minutes depending on network conditions. After the process finished, you'll have a Python environment named `Reno` managed by Anaconda.

To use this environment:
* For Linux users, use `source activate Reno` or `conda activate Reno`
* For Windows users, use `activate Reno`

In most cases the name of the environment will appear in the command line interface. 

To verify the environment is activated, use `python --version` to ensure that you're using Python 3.8. You can also use `which python` (for Linux) or `where python` (for Windows) to see where the Python program locates. It should locate under a directory like `~/anaconda/envs/Reno/bin/` (Note the environment name `Reno` in the path).

### Download Renormalizer and run tests
#### Download Renormalizer
* If you wish to develop Renormalizer, `Git` is a necessary tool. Learning `Git` is beyond the scope of this guide. We assume you already know some basic knowledge of `Git` and how to download Renormalizer using `Git`:
    ```
    git clone https://github.com/jjren/Renormalizer.git
    ```
* If you simply wish to try out Renormalizer, you can [download](https://github.com/shuaigroup/Renormalizer/archive/master.zip) the source files directly.
#### Install dependencies
Renormalizer relies on lots of other Python packages to run. We recommend using `pip` to install these packages. For users in Tsinghua, *tuna* makes life easier by providing the fastest download speed. According to the [official document](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/), run the following command:
```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
to let `pip` download from *tuna*. 

Now, to install the required packages:
1. Enter the `Renormalizer` directory. You should be able to find a `requirements.txt` file.
2. Double check again you're using the correct Python environment
3. install dependencies with:
    ```
    pip install -r requirements.txt
    ```
4. (Optional) For high-performance ground state DMRG, `primme` should also be installed with:
    ```
    pip install primme==3.2.*
    ```
    > Sometimes an error will occur during the installation of `primme`, especially on Windows platform. Please refer to the [primme installation guide](https://github.com/primme/primme) for more details.

#### Run tests
To run tests, use the `pytest` command installed in the previous step in the Renormalizer source code directory:
```
pytest
```
This tool will collect tests in Renormalizer and run them. The tests should run for 10-20 minutes.

> Although the source code of the tests can help you understand how the code works, it is not recommended to start your scientific project by modifying the tests because they are designed for testing and not scientific research. You can firstly read the (test) code and grasp some basic idea on what happens, then write programs using modules in `renormalizer.mps` rather than `renormalizer.spectra` or `renormalizer.transport`. Only use the tests directly when you completely understand each line of the test code.

Finally, to make Python find Renormalizer, add Renormalizer directory to `PYTHONPATH`. For example, if
Renormalizer is installed in `/opt`, you can add the following path to
your `~/.bashrc` and then `source ~/.bashrc`.
```
export PYTHONPATH=/opt/Renormalizer:$PYTHONPATH
```

To ensure the setting is successful, start a Python shell, and try:
```
>>> import renormalizer
```

## GPU-support
### Overview
If you already have a CPU-only version working, the only step for GPU support is to install a new package called [`CuPy`](https://github.com/cupy/cupy), which is a `NumPy` analog for GPU, and then Renormalizer can run with GPU backend.

If the CPU-only version has not been installed, please follow [the steps](#Install-from-scrach) first.

A few things to notice before we start:
* GPU support is only tested under Linux. If you're using Windows or MacOS and something unexpected happens, there could be lots of reasons and fixing the issue might take a while (maybe forever). So we highly recommend a Linux environment for GPU environment.
* Make sure you have a decent NVIDIA GPU. The GPU driver, [CUDA](https://developer.nvidia.com/cuda-toolkit), doesn't support very old GPU, integrated GPU or AMD GPU. You can find the list of CUDA-supported GPUs [here](https://www.geforce.com/hardware/technology/cuda/supported-gpus).
### Install CUDA
Install CUDA following the [official document](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html). All CUDA version later than 8.0 (included) should be OK. Use Google well if any troubles are met. **Be sure to verify the installation**.

> The computer cluster *dirac* of Shuai group has two nodes called `c91` and `c92` each equipped with 4 [V100](https://www.techpowerup.com/gpu-specs/tesla-v100-pcie-32-gb.c3184) GPUs. CUDA has already been installed on the node and you can use the environment by `module` command.

#### (Optional) Install nvtop
[`nvtop`](https://github.com/Syllo/nvtop) is a handy tool to monitor GPU usage. The installation procedure can be quite formidable if you are not familiar with how to use Git and install software from source files. Feel free to skip this step.

> `nvtop` has been installed on `c91` and `c92` on *dirac*.

### Install cupy
After CUDA is installed and verified, you can install `CuPy` using the following commands. Remember when you install something you have to make sure you're in the correct Python environment!
```
(For CUDA 8.0)
$ pip install cupy-cuda80

(For CUDA 9.0)
$ pip install cupy-cuda90

(For CUDA 9.1)
$ pip install cupy-cuda91

(For CUDA 9.2)
$ pip install cupy-cuda92

(For CUDA 10.0)
$ pip install cupy-cuda100
```

> Sometimes you have to install a package without network connection (such as in `c91`). In such cases, you can download the required wheel file under network connection, copy the files to the node and then install with `pip`.

#### Verify cupy
Sometimes `CuPy` can be successfully installed but actually not working. Here is a sample script to test if it's working or not:
```
import cupy as cp
a = cp.random.rand(100, 100)
b = cp.dot(a, a)
print(b.sum())
```
If any error happens, usually it's because the CUDA environment is not correct. Please verify your CUDA installation again and make sure you haven't installed multiple versions of cupy. As a last resort, you can delete the whole Python environment and start over:
```
conda env remove --name Reno
``` 
### Run tests
Just as what we do in the CPU-only version, use
```
pytest
``` 
and wait.

During the test, you can monitor the GPU usage by `nvidia-smi` or `nvtop`. Moderate GPU usage is expected. If GPU usage is zero, probably Renormalizer still runs on CPU. Make sure the python interpreter running the tests can import `cupy`.
