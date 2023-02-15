# About Parareal in Python

This package implements the Parareal algorithm in Python. Parareal is an algorithm for parallel-in-time integration, meaning that it attempts to parallelize the otherwise completely sequential time integration of an ordinary differential equation (ODE). The algorithm works by using two methods to solve the ODE: a cheap and **coarse** method and a more computationally intensive **fine** method. This will only work if the coarse method is good enough so that we can initiate multiple fine integrations from a coarsely estimated intermediate condition.

The body of literature on Parareal is quite extensive. The original method was described by Lions et al. in 2001 [@Lions2001][@AUBANEL2011172].

This Python module implements Parareal as a so-called black-box method. This means that the algorithm doesn't need to know about the details of the simulation. The user writes a script containing a fine and a coarse integrator. The `parareal` module then manages the computation using `dask`.

We have three examples that serve as user documentation.

1. [Dampened harmonic oscillator (simple version)](01-dho-simple.md)
2. [Using HDF5 and MPI (still with harmonic oscillator example)](02-dho-advanced.md)
3. [Pipe flow using OpenFOAM](03-pipeflow.md)


The library is implemented using literate programming, see [Implementation](04-implementation.md).
