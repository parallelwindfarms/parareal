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

# Introduction

## Computational Fluid Dynamics

Computational Fluid Dynamics (CFD) is a branch of physics that uses
computational methods such as numerical analysis to address problems
involving fluid flows. Examples of these problems could be aerodynamics,
weather simulation, acoustics, heat transfer or environmental
engineering. The technological and societal relevance of these problems
is self-evident.

Most fluid dynamics problems involve the resolution of boundary problems
for non-linear partial differential equations. This kind of problems is
notoriously known for the difficulty of their analytical resolution.
Tackling them computationally is also challenging and resource-consuming
[@kundu2010].

A number of software suites is available for solving these problems,
both open source and proprietary. In the present paper we'll focus our
attention in OpenFOAM [@openfoam], one of the most popular software
suites in this field.

## Parallelization of dynamical systems solvers

Resource-consuming computational methods are often a synonym of slow
calculations. Parallel computing, especially since the (relative)
popularization of supercomputers, can be of great help in reducing the
computation time. But in order to get any advantage from
parallelization, we need a parallelizable problem, *i.e.*, a problem
that can be split into smaller, independent problems.

Classical numerical solvers for dynamical systems, such as the
Runge-Kutta algorithm, are not suitable for parallelization. The reason
for this is that each integration step fundamentally depends on the
previous one, making it an inherently serial process. But as we'll see
in section [2.1](#subsec:parareal){reference-type="ref"
reference="subsec:parareal"}, the Parareal algorithm provides a solver
that allows parallelization in the time domain. Under certain
circumstances, the Parareal algorithm leads to a computational
advantage.

## Our contribution

The present work introduces the software solution we developed in order
to apply the Parareal algorithm to calculations with OpenFOAM in a
practical way. In order to attract an audience as big as possible, we
decided to write it in Python, an open-source language with a large and
active users' community. We made it suitable for running on
supercomputers. And last but not least, we followed not only the
principles of FAIR programming [@chue_hong_neil_p_2022_6623556], but
also those of literate programming [@knuth1992] and put great care in
the testing and documentation.

# Approach

## Parareal {#subsec:parareal}

The Parareal algorithm [@Lions2001] allows the parallel implementation
of the time discretization of a partial differential equation. This is
achieved by a combination of a coarse integration algorithm
$\mathcal{G}$ and an independent fine integrator $\mathcal{F}$. Both
integration algorithms provide a way to update the state vector, in the
sense of
$y_{j+1} \approx \mathcal{F}(y_j, t_j, t_{j+1}) \approx \mathcal{G}(y_j, t_j, t_{j+1})$,
where $y_{j+1}$ is the state corresponding to $t_{j+1}$. As the name
suggests, the coarse integrator is computationally cheaper and less
precise in general than the fine one.

The Parareal method is defined by the following iteration:

\begin{equation}\label{eq:parareal}
y_{j+1}^{k+1} = \mathcal{G}(y^{k+1}_j, t_j, t_{j+1}) + \mathcal{F}(y^k_j, t_j, t_{j+1}) - \mathcal{G}(y^k_j, t_j, t_{j+1})
\end{equation}

where $j$ and $k$ are integer indices. $j$ is the corresponding index
for the time discretization $t_j$, while $k$ denotes the iteration
number of Parareal (see figure
[\[fig:parareal\]](#fig:parareal){reference-type="ref"
reference="fig:parareal"}).

As the iteration converges, we expect that
$y^{k+1}_j - y^{k}_j \rightarrow 0$. From
\autoref{eq:parareal}, it follows that
$y^{k+1}_j  \rightarrow \mathcal{F}(y^k_j, t_j, t_{j+1})$ for large
enough $k$, as both instances of $\mathcal{G}$ cancel out. This means
that Parareal converges to the same solution as the fine integrator.

## Parallelization {#subsec:parallelization}

If Parareal's algorithm converges to the same solution as the fine
integrator, why not directly use the latter? The reason is subtle: with
the algorithm split like in \autoref{eq:parareal}, the computationally expensive evaluation of
$\mathcal{F}(y^k_j, t_j, t_{j+1})$ can be performed in parallel on a
number of processing units. By contrast, the dependency of
$y^{k+1}_{j+1}$ on $\mathcal{G}(y^{k+1}_j, t_j, t_{j+1})$ means that the
coarse correction has to be computed in serial order (see figure
[\[fig:parareal\]](#fig:parareal){reference-type="ref"
reference="fig:parareal"}).

::: center
:::

## Convergence criterion {#subsec:convergence}

The minimum acceptable number of iterations of the Parareal algorithm
can in general not be known in advance. The only feasible general
approach is to provide a tolerance, and the iteration stops when
reached. This poses a technical problem to the implementation (see
subsection [3.3](#subsec:futures){reference-type="ref"
reference="subsec:futures"}).

## Regridding

The Parareal algorithm assumes the existence of a cheap but reasonably
accurate *coarse* integrator alongside a more computationally expensive
*fine* integrator. In practice, due to the CFL condition, this means
that the coarse integrator works on a coarser mesh than the fine
integrator.

In order to perform the field additions and subtractions required by
\autoref{eq:parareal}, we need to map the fields between the coarse
and fine meshes. To this goal, we used interpolation. In particular,
OpenFOAM ships with a utility called `mapFields` that solves this
regridding problem efficiently. Our solution builds a wrapper around
this function.

## Windowing

Parareal can give a speedup related to the number of nodes being used.
The number of time slices to use is a parameter that can be played
around with for efficiency and stability. If the stability suffers too
greatly, reducing the spacing between time steps (in Parareal, not in
the fine nor coarse integrators) can help. In the case where the number
of time steps is much greater than the number of nodes available, it
makes sense to have a staged approach. We apply Parareal to a smaller
interval until the results converge, after which we continue in a
sequential manner to the next interval. This approach is known as
windowing, and was also used by @Eghbal2017.

# Implementation

We implemented Parareal in Python and used Dask for orchestrating the
parallel execution. The Parareal algorithm itself has a straightforward
implementation. The recursive definition of Parareal
(\autoref{eq:parareal}) translates into the following Python code for
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
fine-to-coarse grid interpolators used for regridding. Note that here
${\tt y}$ is the list of states from a previous iteration of Parareal,
and ${\tt y_n}$ is the *next* iteration.

By feeding the algorithm Dask delayed functions, a workflow is
automatically generated for parallel execution on any number of
back-ends.

## OpenFOAM

OpenFOAM[^1] is a C++ toolbox for the solution, pre- and post-processing
of computational fluid dynamics problems [@openfoam]. It is free, open
source, and has a large user base across different areas of science and
technology.

## Input/Output

OpenFOAM has support for two types of direct output. One is an ASCII
format with a syntax inherited from C. The other is the same ASCII
format, but with larger data blocks replaced by binary blobs[^2].

We chose to manage IO around the binary file format of OpenFOAM. We
developed a Python module for parsing the ASCII component of these files
and have efficient memory mapped access to the binary field values
inside. This way, we could do scalar arithmetic on the field values by
cloning simulation output and subsequently modifying the field values
inside. Our module is general enough to be able to tackle generic
parsing problems, and we published it independently [@byteparsing2021].

## Dask. Futures vs. promises {#subsec:futures}

As explained in subsection
[2.3](#subsec:convergence){reference-type="ref"
reference="subsec:convergence"}, the stop criterion of our algorithm
relies on a tolerance being met. This means that at some point the
partial results of all cores have to be collected in order to check if
the tolerance has been already met, and if that's the case, stop
iterating.

## Vectors

We define a `Vector` as an object that stores the complete state of the
simulation at any given time. OpenFOAM nativelly stores this information
as a set of files on a folder structure. For the purpose of Parareal we
need to be able to add and subtract vectors. We implemented the class
`Vector` for OpenFOAM snapshots by keeping every instance inside its own
dedicated case folder. When we need to add two vectors, we clone the
first and then modify the field values of the cloned instance in place.
Cloning a vector amounts to copying the basic folder structure of an
OpenFOAM case directory, together with the time directory containing the
field values for that snapshot.

## Integrators

We need to provide two integrators to the `parareal` function, along
with two functions that can map vectors from a coarse to a fine grid and
vice-versa. In this instance we have two integrators that both call the
`pimpleFoam` executable, but with different time steps (and on different
grids, but for clarity that is left out of this example):

``` {.python language="Python"}
from dask import delayed
from pintFoam.foam import foam

@delayed
def fine(n, x, t_0, t_1):
    return foam("pimpleFoam", 0.1, x, t_0, t_1)

@delayed
def coarse(n, x, t_0, t_1):
    return foam("pimpleFoam", 1.0, x, t_0, t_1)
```

By providing the `parareal` function with Dask delayed functions, we
automatically build a workflow of the algorithm, ready for parallel
execution. Passing these functions to `parareal` gives us the workflow
shown in
Figure [\[fig:parareal-graph\]](#fig:parareal-graph){reference-type="ref"
reference="fig:parareal-graph"}.

[]{#fig:parareal-graph label="fig:parareal-graph"}
![image](figs/parareal-graph.pdf){width="\\textwidth"}

-   Python + Dask

-   Main script

-   Dependency graph

# Discussion and performance

-   Setup

-   IO

-   Parareal

## Test problem 1: 2D laminar flow inside a pipe

Our first case study will be that of a two-dimensional laminar flow
inside a pipe. This is the classical textbook example.

## Test problem 2: 2D laminar flow around a cylinder

Our case study will be that of a two-dimensional laminar flow around a
cylinder. Contrary to the first test problem, the flow around a cylinder
doesn't converge to a stationary solution if the Reynolds' number is
high enough.

[^1]: Acronym for \"Open-source Field Operation And Manipulation\"

[^2]: There is also premature support for the Adios2 file format that is
    better suited for HPC applications. However, the support for Adios
    is not yet mature enough for general adoption.
