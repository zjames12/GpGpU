
# GpGpU

GpGpU is an R package for fast approximate Gaussian process computation.
The package includes implementations of the Vecchia's (1988) original
approximation, as well as several updates to it, including the reordered
and grouped versions of the approximation outlined in Guinness (2018).

GpGpU supports the use of CUDA enabled GPUs. GPU acceleration is currently available for calculating the sparse inverse Cholesky matrix implied by Vecchia's with the exponential isotropic covariance function.

## Installing

The package can be installed on the command line.

```Shell
$ R CMD build GpGpU
$ R CMD INSTALL GpGpU_0.4.0.tar.gz
```

Building this package requires modifying the Makevars file to match system locations of your R and CUDA libraries. GpGpU is not supported on Windows.

## Installing On a G2 Account

Several additional steps are required to run GpGpU on a G2 account. First add conda to your path if have not already done so.

```Shell
netid@g2-login-01:~$ /share/apps/anaconda3/2021.05/bin/conda init
```

Next create a virtual environment with Anaconda that includes R.

```{r}
netid@g2-login-01:~$ conda create -n r-env r-essentials r-base
netid@g2-login-01:~$ conda activate r-env
```

Clone GpGpU and install the package.

```Shell
netid@g2-login-01:~$ git clone https://github.com/zjames12/GpGpU.git
netid@g2-login-01:~$ R CMD build GpGpU
netid@g2-login-01:~$ R CMD INSTALL GpGpU_0.4.0.tar.gz
```

 Any R scripts submitted to a compute node can now use GpGpU. Additonal resources can be found below:

- [Using a virtual environment in the G2 cluster](https://it.coecis.cornell.edu/researchit/g2cluster/g2-virtual-environments/)
- [Using R language with Anaconda](https://docs.anaconda.com/free/anaconda/packages/using-r-language/#:~:text=When%20using%20conda%20to%20install,type%20conda%20install%20r%2Drjava%20.)
