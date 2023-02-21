---
title: 'Parallel-in-time integrator. With application to OpenFOAM'
tags:
  - Python
  - modeling
  - Fluid dynamics
authors:
  - name: Johan Hidding
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: true
    affiliation: 1
    orcid: 0000-0002-7550-1796
  - name: Pablo Rodríguez-Sánchez
    orcid: 0000-0002-2855-940X
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Netherlands eScience Center
   index: 1
date: 25 October 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Classical numerical solvers for dynamical systems, such as the
Runge-Kutta algorithm, are not suitable for parallelization. The reason
for this is that each integration step fundamentally depends on the
previous one, making it an inherently serial process. But as we'll see
below, the Parareal algorithm [@Lions2001] provides a solver
that allows parallelization in the time domain. The Parareal algorithm
is known to lead to a computational advantage under certain circumstances
[@AUBANEL2011172].

The present work introduces the software solution we developed in order
to apply the Parareal algorithm to different dynamical problems in a practical way.
In order to attract an audience as big as possible, we
decided to write it in Python, an open-source language with a large and
active users' community. We made it suitable for running on
supercomputers. And last but not least, we followed not only the
principles of FAIR programming [@chue_hong_neil_p_2022_6623556], but
also those of literate programming [@knuth1992] and put great care in
the testing and documentation.

# Statement of need

Our motivation for creating this package was the resolution of Computational
Fluid Dynamics (CFD) problems, in particular, with OpenFOAM[^1] [@openfoam], one of the most popular software
suites in the field of CFD.

CFD is a branch of physics that uses
computational methods such as numerical analysis to address problems
involving fluid flows. Examples of these problems could be aerodynamics,
weather simulation, acoustics, heat transfer or environmental
engineering.

Most fluid dynamics problems involve the resolution of boundary problems
for non-linear partial differential equations. This kind of problems is
notoriously known for the difficulty of their analytical resolution.
Tackling them computationally is also challenging and resource-consuming
[@kundu2010]. Parallelization in time is one of the proposed methods to
speed-up this kind of computations.

Being a relatively exotic algorithm, Parareal is not natively supported
by OpenFOAM. Last but not least, our package allows for using generic
solvers, making it a good team mate for other dynamical systems solvers.

<!-- The commented lines below are more interesting to researchers than to software engineers. I suggest removing -->
<!-- Most accepted approaches for achieving parallel computation in CFD involve subdividing the domain into many components for which solving the system of equations are relatively independent. If we want to add more nodes to our computation, we need to subdivide the work into smaller pieces. The problem is that with smaller sub-domains, the communication overhead increases, until adding more processors does not give any speed-up. Should we want to get our results faster, we need to look for alternative methods to speed up our calculations. One proposed method is to go parallel-in-time. -->

# Notes on architecture

## Implementation

We implemented Parareal in Python and used Dask for orchestrating the
parallel execution. The Parareal algorithm itself has a straightforward
implementation. The recursive definition of Parareal
(see \autoref{eq:parareal} in Appendix) translates into the following Python code for
a single iteration:

``` {.python language="Python"}
def parareal(coarse, fine, c2f, f2c):
    def f(y, t):
        y_n = [None] * t.size
        y_n[0] = y[0]
        for j in range(1, t.size):
            y_n[i] = c2f(coarse(f2c(y_n[j-1]), t[j-1], t[j])) \
                   + fine(y[j-1], t[j-1], t[j]) \
                   - c2f(coarse(f2c(y[j-1]), t[j-1], t[j]))
        return y_n
    return f
```

Where $\tt coarse$ and $\tt fine$ are, respectively, the coarse and fine
integrators, and $\tt c2f$ and $\tt f2c$ the coarse-to-fine and
fine-to-coarse grid interpolators used for regridding[^2]. Note that here
${\tt y}$ is the list of states from a previous iteration of Parareal,
and ${\tt y_n}$ is the *next* iteration.

By feeding the algorithm Dask delayed functions, a workflow is  <!-- TODO: Consider adding a citation to Dask delayed functions -->
automatically generated for parallel execution on any number of
back-ends.

See examples in next section.

## Input/Output in OpenFOAM

We define a `Vector` as an object that stores the complete state of the
simulation at any given time. OpenFOAM nativelly stores this information
as a set of files on a folder structure.

OpenFOAM supports two types of direct output. One is an ASCII
format with a syntax inherited from C. The other is the same ASCII
format, but with larger data blocks replaced by binary blobs.

We chose to manage IO around the binary file format of OpenFOAM. We
developed a Python module for parsing the ASCII component of these files
and have efficient memory mapped access to the binary field values
inside. Our module is general enough to be able to tackle generic
parsing problems, and we published it independently [@byteparsing2021].

Apart from loading OpenFOAM data into memory, we also need to be able to
add and subtract vectors. The class `Vector` takes care of all of this
in a practical manner[^3].

<!-- TODO: consider keeping only one example -->
## Test problem 1: 2D laminar flow inside a pipe

Our first case study will be that of a two-dimensional laminar flow
inside a pipe. This is the classical textbook example.

## Test problem 2: 2D laminar flow around a cylinder

Our case study will be that of a two-dimensional laminar flow around a
cylinder. Contrary to the first test problem, the flow around a cylinder
doesn't converge to a stationary solution if the Reynolds' number is
high enough.

# Acknowledgements
This project was supported by funding from the Netherlands eScience Center and NWO as part of the Joint Call for Energy Research, Project Number CSER.JCER.025. We also want to acknowledge Dr. Nicolas Renaud for his support and suggestions.

# References

[^1]: Acronym for \"Open-source Field Operation And Manipulation\".
OpenFOAM is a C++ toolbox for the solution, pre- and post-processing
of computational fluid dynamics problems. It is free, open
source, and has a large user base across different areas of science and
technology.

[^2]: The Parareal algorithm assumes the existence of a cheap but reasonably
accurate *coarse* integrator alongside a more computationally expensive
*fine* integrator. In practice this means that the coarse integrator
works on a coarser mesh than the fine integrator (for details, see
Courant–Friedrichs–Lewy condition, often referred to as CFL).

[^3]: When we need to add two vectors, we clone the
first and then modify the field values of the cloned instance in place.
Cloning a vector amounts to copying the basic folder structure of an
OpenFOAM case directory, together with the time directory containing the
field values for that snapshot.
