# GpGpU

GpGpU is an R package for computing approximate Gaussian process models on a GPU. The package includes all the methods in the [GpGp](https://cran.r-project.org/web/packages/GpGp/index.html) package, with additional support for fitting with a CUDA GPU. Several covariance functions are supported.

## Installing

The package can be installed on the command line.

```Shell
$ R CMD build GpGpU
$ R CMD INSTALL GpGpU_0.4.0.tar.gz
```

Building this package requires modifying the Makevars file to match system locations of your R and CUDA libraries. GpGpU is not supported on Windows.

## Usage

All GpGp code is compatible with GpGpU. A new optional argument in the fit function makes fitting models on the GPU easy.

```R
y <- argo2016$temp100
locs <- argo2016[c("lat","lon")]
m <- fit_model(y, locs, covfun_name = "exponential_isotropic", gpu = T)

```

GpGpU does not support grouping of observations. The following covariance functions from GpGp are currently supported.

- exponential_isotropic
- exponential_scaledim
- exponential_spacetime
- exponential_spheretime
