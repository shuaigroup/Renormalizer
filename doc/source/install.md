# Installation guide
This is a step by step guide on how to prepare an environment for Renormalizer from scratch. Although the name of this guide is "installation guide", we actually won't "install" Renormalizer as a standalone package that can be imported anywhere. What we achieve in this guide is to put source files of Renormalizer in your computer and prepare a Python environment that can run it.

Renormalizer runs on both CPU and GPU. It is recommended to install CPU-only version first, then GPU support (if possible).

The CPU-only version runs on Windows/MacOSX/Linux, whereas the GPU support is only tested on Linux. 

As operations on MacOSX are similar to those on Linux, from now on we'll only talk about Windows and Linux.

## CPU-only version installation

### Python environment
There are various ways to prepare a Python environment for Renormalizer. Here we recommend using Anaconda (both 2 or 3 are ok) as a package and environment manager.
#### Why you need Anaconda when you already have Python
Anaconda manages *environments*. If you use Python frequently enough, one day you'll face problems similar to this: project A runs on Python 3.5 but not Python 3.6, project B runs only on Python 3.6, how could I develop project A and project B on the same machine? Anaconda can help you solve these problems with ease.

Another common issue about Python is that its environment screws up quite frequently. You may accidentally install multiple versions of the same package or delete some critical packages and then you'll get a strange error importing the package you want. Find out what went wrong can take hours and you just wish you can reinstall Python. Anaconda can help you reinstall Python clean and easy with 2-3 commands.

Anaconda has lots of other cool features, you'll learn more while you're using it.

If you find the installation of Anaconda annoying and you already have Python 3.6, in theory, you can skip the installation and use your Python directly. But if something went wrong and your Python couldn't work anymore, you'll probably feel that using Anaconda is a good idea.

#### Install Anaconda
You can skip this step if you already have access to Anaconda (which means you can use `conda` command in your command line). For Linux users, don't worry if you don't have the root privilege -- installing Anaconda doesn't require root.

> A reminder to dirac2 users: both Anaconda2 and Anaconda3 have been installed on the cluster. Try `module load anaconda/3`.

For students in Tsinghua, Anaconda can be downloaded from *[tuna](https://tuna.tsinghua.edu.cn/)* free of network charge. You can simply think of *tuna* as a cloud disk inside Tsinghua.

There are lots of links in the [download page](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/). You need to choose a correct version for your operating system and preferably a relatively newer version (at the bottom of the page).

For students outside Tsinghua, please download Anaconda [here](https://www.anaconda.com/)

For Linux users, use `bash` to run the installation script.

After you have installed Anaconda, you can test your installation using `conda` command. Anaconda has a graphic user interface for MacOSX and Windows, however we recommend using a command-line interface. If your command line can't find `conda`, please try to locate where `conda` is installed and add `conda` (`conda.exe` for Windows) to your system environment `PATH`, or you can use `conda prompt` if it is installed.

#### Create an environment
If you have installed Anaconda you should now be able to use Python in your command line. This Python is shipped with Anaconda and we recommend **not** using it for any particular Python project. The best practice is to create an environment for every project so that they don't mess up. 

Students inside Tsinghua can use *tuna* to speed up preparing the environment by providing maximum download speed (and again saves network charge). See [this page](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) for the official document. In short, use the following command:
``` 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/    
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/    
conda config --set show_channel_urls yes    
``` 
to let Anaconda know it can download resources it needs from *tuna*.

In this step, we're going to create a Python environment for Renormalizer using Anaconda. Two things must be taken into consideration while creating a Python environment:
* Python version. **Python version higher than Python 3.5 is required for Renormalizer**. We recommend using Python 3.6 as it is more stable than Python 3.7.
    > Renormalizer uses *type annotation* like `a: int = 3` or `b: float = 3.0` which helps to make code clearer but requires Python 3.6 or higher. For more info about *type annotation*, see the [typing module](https://docs.python.org/3/library/typing.html) or [PEP484](https://www.python.org/dev/peps/pep-0484/).
* Name of the environment. Choose anything you like. **In this guide we are going to use `Renormalizer` as it is the name of the project**.

With Python version and name of the environment determined, create the environment using the following command (Note the environment name `Renormalizer`)
```
conda create -n Renormalizer python=3.6 -y
```
This might take 1 to 20 minutes depending on network conditions. After the process finished, you'll have a Python environment named `Renormalizer` managed by Anaconda.

To use this environment:
* For Linux users, use `source activate Renormalizer`
* For Windows users, use `activate Renormalizer`

You may notice that the name of your environment is added to your command line interface. 

To verify the environment is activated, use `Python --version` to ensure that you're using Python 3.6. You can also use `which python` (for Linux) or `where python` (for Windows) to see where your Python program locates. It should locate under a directory like `~/anaconda/envs/Renormalizer/bin/` (Note the environment name `Renormalizer` in the path).

### Download Renormalizer and run tests
#### Download Renormalizer
* If you wish to develop Renormalizer, `Git` is a necessary tool. Learning `Git` is beyond the scope of this guide. We assume you already know some basic knowledge of `Git` and how to download Renormalizer using `Git`:
    ```
    git clone https://github.com/jjren/Renormalizer.git
    ```
* If you simply wish to try out Renormalizer, you can [download](https://github.com/jjren/Renormalizer/archive/master.zip) the source files directly.
#### Install dependencies
Renormalizer relies on lots of other Python packages to run. We recommend using `pip` to install these packages. For students in Tsinghua, *tuna* makes life easier by providing the fastest download speed. According to the [official document](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/), we only need to do:
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
4. install `qutip` (which depends on the dependencies) with:
    ```
    pip install qutip==4.3.1
    ```
#### Run tests
To run tests, simply use the `pytest` command installed in the previous step in your Renormalizer source code directory:
```
pytest
```
This tool will collect tests in Renormalizer and run them. The tests should run for 20-30 minutes depending on the computational power of your platform.

> Although the tests can help you understand how the code works, it is not recommended to start your scientific project by modifying the tests because they are designed for testing and not scientific research. You can firstly read the (test) code and grasp some basic idea on what happens, then write programs using modules in `Renormalizer.mps` rather than `Renormalizer.spectra` or `Renormalizer.transport`. Only use the tests directly when you completely understand each line of the test code.

## GPU-support
### Overview
If you already have a CPU-only version working, add GPU-support to Renormalizer is not complex. You don't have to reinstall anything, you simply need to install a new package called [`cupy`](https://github.com/cupy/cupy), which is similar to a `NumPy` for GPU, and then Renormalizer can run with GPU backend.

If you currently don't have a CPU-only version, please follow [the steps](#cpu-only-version-installation) first. Of course, if you are familiar with Python and Renormalizer, you can mix the two separate guides into one.

A few things to notice before we start:
* GPU support is only tested under Linux. If you're using Windows or MacOS and something unexpected happens, there could be lots of reasons and fixing the issue might take a while (maybe forever). So we highly recommend a Linux environment for GPU-support.
* Make sure you have a decent NVIDIA GPU. We have to install [CUDA](https://developer.nvidia.com/cuda-toolkit) for GPU support, but it doesn't support very old GPU, integrated GPU or AMD GPU. You can find the list of CUDA-supported GPU [here](https://www.geforce.com/hardware/technology/cuda/supported-gpus).
### Install CUDA
Install CUDA following the [official document](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html). All CUDA version later than 8.0 (included) should be OK. Use Google well if you met any trouble while following the steps in the document. **Be sure to verify your installation**.

> The computer cluster of our group has one node called `c91` equipped with 2 [V100](https://www.techpowerup.com/gpu-specs/tesla-v100-pcie-32-gb.c3184) GPU. CUDA has already been installed on the node and you can use the environment by `module` command.

#### (Optional) Install nvtop
[`nvtop`](https://github.com/Syllo/nvtop) is a handy tool to monitor GPU usage. The installation procedure can be quite formidable if you are not familiar with how to use Git and install software from source files. Feel free to skip this step.
### Install cupy
After CUDA is installed and verified, you can install `cupy` using the following commands. Remember when you install something you have to make sure you're in the correct Python environment!
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
Sometimes `cupy` can be successfully installed but actually not working. You can run a simple script to test if it's working or not:
```
import cupy as cp
a = cp.random.rand(100, 100)
b = cp.dot(a, a)
print(b.sum())
```
Of course, you can run anything else to test the functions. If any error happens, usually it's because the CUDA environment is not correct. Please verify your CUDA installation again and make sure you haven't installed multiple versions of cupy. As a last resort, you can delete the whole Python environment and start over:
```
conda env remove Renormalizer
``` 
It won't take lots of time if you're using *tuna*.
### Run tests
Just as what we do in the CPU-only version, use
```
pytest
``` 
and wait. 

During the test, you can monitor the GPU usage by `nvidia-smi` or `nvtop`. Moderate GPU usage is expected. If GPU usage is zero, probably Renormalizer still runs on CPU. Make sure the python interpreter running the tests can import `cupy`.

If the test is using GPU and passed, congrats! You can now enjoy 10x performance boost on some of the Renormalizer calculations.

## Further readings
Currently, we don't have detailed documentation on how to use the code and the best way to learn how the code works is by reading the source code.

Some notes on [how to run the code](https://github.com/jjren/Renormalizer/wiki/How-to-use).

Note: **Renormalizer is still under heavy development. API changes come without any prior deprecation.**
