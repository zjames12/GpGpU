
# GpGpU

GpGpU is an R package for fast approximate Gaussian process computation.
The package includes implementations of the Vecchia's (1988) original
approximation, as well as several updates to it, including the reordered
and grouped versions of the approximation outlined in Guinness (2018).

GpGpU supports the use of CUDA enabled GPUs.

## Installing

The package can be installed on the command line

```{r}
$ R CMD build GpGpU
$ R CMD INSTALL GpGpU_0.4.0.tar.gz
```

Build this package requires modifying the Makevars file. GpGpU is not supported on Windows.
