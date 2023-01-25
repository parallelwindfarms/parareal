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

## Statement of need

Our motivation for creating this package was the resolution of Computational
Fluid Dynamics (CFD) problems; in particular, with OpenFOAM [@openfoam], one of the most popular software
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
number of Parareal (see \autoref{fig:parareal}).

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
coarse correction has to be computed in serial order (see \autoref{fig:parareal}).

\begin{figure}
\begin{center}
\begin{tikzpicture}
	% Grid
  	\draw[thin,gray!40] (-0.25,-0.25) grid (4.0,3.5);

  	% Axes
    \draw[<->] (0,0)--(4.5, 0) node[right]{$j$};
  	\draw[<->] (0,0)--(0, 3.5) node[above]{$k$};
  	
  	% Initial conditions
  	\filldraw [blue!40] (0,0) circle (4pt);
  	\filldraw [blue!40] (0,1) circle (4pt);
  	\filldraw [blue!40] (0,2) circle (4pt);
  	\filldraw [blue!40] (0,3) circle (4pt);
  	
  	% First coarse integration
  	\filldraw [red!40] (1,0) circle (4pt);
  	\filldraw [red!40] (2,0) circle (4pt);
  	\filldraw [red!40] (3,0) circle (4pt);
  	\filldraw [red!40] (4,0) circle (4pt);

	% Iterator
  	\draw[->, ultra thick, red,  arrows={-latex}]  (2,2) -- (1,2) node[midway,above] {$\mathcal{G}$};
  	\draw[->, ultra thick, red,  arrows={-latex}]  (2,2) -- (1,1) node[below] {$\mathcal{F-G}$};
  	\filldraw [color=red, fill=white] (2,2) circle (4pt);
  	
\end{tikzpicture}
\end{center}
\caption{\label{fig:parareal} Discretization diagram corresponding to equation \ref{eq:parareal}. 
The $j$ index corresponds to each value of time $t_j$. The $k$ index corresponds to each iteration of Parareal. 
The blue dots at $j=0$ correspond to the initial condition; all of them are identical and are provided as part of the problem definition.
The red dots at $k=0$ are obtained by applying the coarse integrator $\mathcal{G}$ once.
The discretization diagram, in red, indicates that the computation of $y^k_{j+1}$ requires both $y^{k-1}_{j}$ and $y^k_{j}$ as an input.
Visualized like this, it is apparent that sweeping the integrator first horizontally and then vertically, we can compute all the values of $y^k_j$.
Note that the contribution of the fine, computationally-expensive integrator $\mathcal{F}$ involves only the values calculated in the previous iteration, and can thus be computed in parallel by assigning a node to each time window.
}
\end{figure}

## Convergence criterion {#subsec:convergence}

The minimum acceptable number of iterations of the Parareal algorithm
can in general not be known in advance. The only feasible general
approach is to provide a tolerance, and the iteration stops when
reached. This poses a technical problem to the implementation (see
more on subsection about Dask).

## Regridding

The Parareal algorithm assumes the existence of a cheap but reasonably
accurate *coarse* integrator alongside a more computationally expensive
*fine* integrator. In practice this means that the coarse integrator
works on a coarser mesh than the fine integrator (for details, see
Courant–Friedrichs–Lewy condition, often referred to as CFL).

In order to perform the field additions and subtractions required by
\autoref{eq:parareal}, we need to map the fields between the coarse
and fine meshes. To this goal, we used interpolation. In particular,
OpenFOAM ships with a utility called `mapFields` that solves this
regridding problem efficiently. Our solution builds a wrapper around
this function.

<!-- TODO: consider removing -->
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

By feeding the algorithm Dask delayed functions, a workflow is <!-- TODO: Consider adding a citation to Dask delayed functions-- > 
automatically generated for parallel execution on any number of
back-ends.

## OpenFOAM

OpenFOAM[^1] is a C++ toolbox for the solution, pre- and post-processing
of computational fluid dynamics problems [@openfoam]. It is free, open
source, and has a large user base across different areas of science and
technology.

### Input/Output

OpenFOAM supports two types of direct output. One is an ASCII
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
<!-- TODO: either complete or remove this section -->
As explained in the subsection about convergence, the stop criterion of our algorithm
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
shown in \autoref{fig:parareal-graph}.

![image.\label{fig:parareal-graph}](figs/parareal-graph.pdf){width="\\textwidth"}

-   Python + Dask

-   Main script

-   Dependency graph

<!-- TODO: consider removing -->
# Discussion and performance

-   Setup

-   IO

-   Parareal

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

[^1]: Acronym for \"Open-source Field Operation And Manipulation\"

<!-- TODO: Consider removing -->
[^2]: There is also premature support for the Adios2 file format that is
    better suited for HPC applications. However, the support for Adios
    is not yet mature enough for general adoption.
