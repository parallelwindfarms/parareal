# Implementation

```python
# file="parareal/__init__.py"
import subprocess

from .tabulate_solution import tabulate
from .parareal import parareal
from . import abstract

__all__ = ["tabulate", "parareal", "schedule", "abstract"]
```

## Components
We may present the Parareal algorithm in abstract terms, and match those terms with corresponding type definitions in Python.

We need to define the following:

### Vector
A `Vector` is an object that represents the state of a solution at any one time. On this state we need to be able to do addition, subtraction and scalar multiplication, in order to perform the Parareal algorithm.

### Solution
A `Solution` is a function that takes an initial `Vector`, a time `t_0` and a time `t`, returning the state `Vector` at time `t`.

### Mapping
A `Mapping` is a function from one state `Vector` to another, for example a mapping from a coarse to a fine mesh or vice-versa.

### Fine Solution
The *fine* solution is the solution at the desired resolution. If we were not doing parallel-in-time, this would be the integrator to get at the correct result. We may also use the fine solution to find a ground thruth in testing the Parareal solution.

### Coarse Solution
The *coarse* solution is the solution that is fast but less accurate.

``` {.python title="parareal/abstract.py"}
from __future__ import annotations
from typing import (Callable, Protocol, TypeVar, Union)

<<abstract-types>>
```

### Vector
We have an ODE in the form

$$y' = f(y, t).$$

Here $y$ can be a scalar value, a vector of values (say a `numpy` array), or any expression of *state*. A naive implementation of an ODE integrator would be

$$y_{n+1} = y_{n} + \Delta t f(y_{n}, t).$$

+@eq:euler-method is known as the *forward Euler method*. We can capture the *state* $y$ in an abstract class we'll call `Vector`. We chose this name because we expect this objects to share (some of) the arithmetic properties of mathematical vectors. Namely, we want to be able to add, subtract and scale them. The chunk below states this need of a basic arithmetic in the form of abstract methods.

``` {.python title="#abstract-types"}
TVector = TypeVar("TVector", bound="Vector")

class Vector(Protocol):
    def __add__(self: TVector, other: TVector) -> TVector:
        ...

    def __sub__(self: TVector, other: TVector) -> TVector:
        ...

    def __mul__(self: TVector, other: float) -> TVector:
        ...

    def __rmul__(self: TVector, other: float) -> TVector:
        ...

```

_We don't actually need to implement these methods right now. All this is saying, is that any type that has these methods defined can stand in for a `Vector`._

Note that we don't make a formal distinction here between a state vector and a vector representing a change in state.


``` {.python title="#abstract-types"}
Mapping = Callable[[TVector], TVector]
```

### Problem
An ODE is then given as a function taking a `Vector` (the state $y$) and a `float` (the time $t$) returning a `Vector` (the derivative $y' = f(y,t)$ evaluated at $(y,t)$). We define the type `Problem`:

``` {.python title="#abstract-types"}
Problem = Callable[[TVector, float], TVector]
```

In mathematical notation the snippet above means:

$${\rm Problem} : (y, t) \to f(y, t) = y'$$

### Solution
If we have a `Problem`, we're after a `Solution`: a function that, given an initial `Vector` (the initial condition $y_0$), initial time ($t_0$) and final time ($t$), gives the resulting `Vector` (the solution, $y(t)$ for the given initial conditions).

``` {.python title="#abstract-types"}
Solution = Union[Callable[[TVector, float, float], TVector],
                 Callable[..., TVector]]
```

Those readers more familiar with classical physics or mathematics may notice that our `Problem` object corresponds with the function $f$. The `Solution` object, on the other hand, corresponds with the evolution operator $\phi$:

$${\rm Solution} : (y_0, t_0; t) \to \phi(y_0, t_0; t) = y.$${#eq:solution}

Intuitively, $\phi$ represents any method that solves (even approximately) our initial value problem.

#### Example
An example of a `Problem` would be the function,

$$f(y, t) = r y,$$

in which case the corresponding `Solution` is,

$$\phi(y_0, t_0; t) = y_0 e^{r(t - t_0)}.$$

### Solver
The challenge is, of course, to find a way of transforming a `Problem` into a `Solution`. This is what integration algorithms, or *solvers* do:

$${\rm Solver} : {\rm Problem} \to {\rm Solution}.$$

If we look a bit closely at the definitions of `Problem` and `Solution` we'll notice that a solver is indeed a functional that accepts functions of $(y,t)$ as an input and returns functions of $(y_0, t_0, t)$ as an output.

An example of such a solver is the forward Euler method (+@eq:euler-method), that can be implemented as:

``` {.python title="parareal/forward_euler.py"}
from .abstract import (Vector, Problem, Solution)

def forward_euler(f: Problem) -> Solution:
    """Forward-Euler solver."""
    def step(y: Vector, t_0: float, t_1: float) -> Vector:
        """Stepping function of Euler method."""
        return y + (t_1 - t_0) * f(y, t_0)
    return step
```

Any existing solution can be iterated over to provide a solution over a larger time interval. The `iterate_solution` function runs a given solution with a step-size fixed to $\Delta t = h$.

<!--$${\rm Iter}[S, h]\Big|_{t_0, y = y}^{t_1} = \begin{cases}-->
<!--y & t_0 = t_1 \\-->
<!--{\rm Iter}[S, h]\big|_{t_0 + h, y = S(y, t_0, t_0 + h)}^{t_1} & {\rm otherwise}-->
<!--\end{cases}.$$-->

``` {.python title="parareal/iterate_solution.py"}
from .abstract import (Vector, Solution)
import numpy as np
import math

def iterate_solution(step: Solution, h: float) -> Solution:
    def iter_step(y: Vector, t_0: float, t_1: float) -> Vector:
        """Stepping function of iterated solution."""
        n = math.ceil((t_1 - t_0) / h)
        steps = np.linspace(t_0, t_1, n + 1)
        for t_a, t_b in zip(steps[:-1], steps[1:]):
            y = step(y, t_a, t_b)
        return y
    return iter_step
```


#### Numeric solution
To plot a `Solution`, we need to tabulate the results for a given sequence of time points.

``` {.python title="parareal/tabulate_solution.py"}
from .abstract import (Solution, Vector)
from typing import (Sequence, Any)
import numpy as np

Array = Any

def tabulate(step: Solution, y_0: Vector, t: Array) -> Sequence[Vector]:
    """Tabulate the step-wise solution, starting from `y_0`, for every time
    point given in array `t`."""
    if isinstance(y_0, np.ndarray):
        return tabulate_np(step, y_0, t)

    y = [y_0]
    for i in range(1, t.size):
        y_i = step(y[i-1], t[i-1], t[i])
        y.append(y_i)
    return y

<<tabulate-np>>
```

In the case that the `Vector` type is actually a numpy array, we can specialize the `tabulate` routine to return a larger array.

``` {.python title="#tabulate-np"}
def tabulate_np(step: Solution, y_0: Array, t: Array) -> Array:
    y = np.zeros(dtype=y_0.dtype, shape=(t.size,) + y_0.shape)
    y[0] = y_0
    for i in range(1, t.size):
        y[i] = step(y[i-1], t[i-1], t[i])
    return y
```

``` {.python title="build/plot-harmonic-oscillator.py"}
import matplotlib.pylab as plt
import numpy as np

from pintFoam.parareal.harmonic_oscillator import \
    ( harmonic_oscillator, underdamped_solution )
from pintFoam.parareal.forward_euler import \
    ( forward_euler )
from pintFoam.parareal.tabulate_solution import \
    ( tabulate )

omega_0 = 1.0
zeta = 0.5
f = harmonic_oscillator(omega_0, zeta)
t = np.linspace(0.0, 15.0, 100)
y_euler = tabulate(forward_euler(f), np.r_[1.0, 0.0], t)
y_exact = underdamped_solution(omega_0, zeta)(t)

plt.plot(t, y_euler[:,0], color='slateblue', label="euler")
plt.plot(t, y_exact[:,0], color='orangered', label="exact")
plt.plot(t, y_euler[:,1], color='slateblue', linestyle=':')
plt.plot(t, y_exact[:,1], color='orangered', linestyle=':')
plt.legend()
plt.savefig("docs/img/harmonic.svg")
```

We can compare the results from the numeric integration with the exact solution.

``` {.make .figure target=img/harmonic.svg}
Damped harmonic oscillator
---
$(target): build/plot-harmonic-oscillator.py
> python $<
```

## Parareal

From Wikipedia:

> Parareal solves an initial value problem of the form
>
> $$\dot{y}(t) = f(y(t), t), \quad y(t_0) = y_0 \quad \text{with} \quad t_0 \leq t \leq T.$$
>
> Here, the right hand side $f$ can correspond to the spatial discretization of a partial differential equation in a method of lines approach.
> Parareal now requires a decomposition of the time interval $[t_0, T]$ into $P$ so-called time slices $[t_j, t_{j+1}]$ such that
>
> $$[t_0, T] = [t_0, t_1] \cup [t_1, t_2] \cup \ldots \cup [t_{P-1}, t_{P} ].$$
>
> Each time slice is assigned to one processing unit when parallelizing the algorithm, so that $P$ is equal to the number of processing units used for Parareal.
>
> Parareal is based on the iterative application of two methods for integration of ordinary differential equations. One, commonly labelled ${\mathcal {F}}$, should be of high accuracy and computational cost while the other, typically labelled ${\mathcal {G}}$, must be computationally cheap but can be much less accurate. Typically, some form of Runge-Kutta method is chosen for both coarse and fine integrator, where ${\mathcal {G}}$ might be of lower order and use a larger time step than ${\mathcal {F}}$. If the initial value problem stems from the discretization of a PDE, ${\mathcal {G}}$ can also use a coarser spatial discretization, but this can negatively impact convergence unless high order interpolation is used. The result of numerical integration with one of these methods over a time slice $[t_{j}, t_{j+1}]$ for some starting value $y_{j}$ given at $t_{j}$ is then written as
>
> $$y = \mathcal{F}(y_j, t_j, t_{j+1})\ {\rm or}\ y = \mathcal{G}(y_j, t_j, t_{j+1}).$$
>
> Serial time integration with the fine method would then correspond to a step-by-step computation of
>
> $$y_{j+1} = \mathcal{F}(y_j, t_j, t_{j+1}), \quad j=0, \ldots, P-1.$$
>
> Parareal instead uses the following iteration
>
> $$y_{j+1}^{k+1} = \mathcal{G}(y^{k+1}_j, t_j, t_{j+1}) + \mathcal{F}(y^k_j, t_j, t_{j+1}) - \mathcal{G}(y^k_j, t_j, t_{j+1}),\\ \quad j=0, \ldots, P-1, \quad k=0, \ldots, K-1,$$
>
> where $k$ is the iteration counter. As the iteration converges and $y^{k+1}_j - y^k_j \to 0$, the terms from the coarse method cancel out and Parareal reproduces the solution that is obtained by the serial execution of the fine method only. It can be shown that Parareal converges after a maximum of $P$ iterations. For Parareal to provide speedup, however, it has to converge in a number of iterations significantly smaller than the number of time slices, that is $K \ll P$.
>
> In the Parareal iteration, the computationally expensive evaluation of $\mathcal{F}(y^k_j, t_j, t_{j+1})$ can be performed in parallel on $P$ processing units. By contrast, the dependency of $y^{k+1}_{j+1}$ on $\mathcal{G}(y^{k+1}_j, t_j, t_{j+1})$ means that the coarse correction has to be computed in serial order.

Don't get blinded by the details of the algorithm. After all, everything boils down to an update equation that uses a state vector $y$ to calculate the state at the immediately next future step (in the same fashion as equation +@eq:euler-method did). The core equation translates to:

``` {.python title="#parareal-core-1"}
y_n[i] = coarse(y_n[i-1], t[i-1], t[i]) \
       + fine(y[i-1], t[i-1], t[i]) \
       - coarse(y[i-1], t[i-1], t[i])
```

If we include a `Mapping` between fine and coarse meshes into the equation, we get:

``` {.python title="#parareal-core-2"}
y_n[i] = c2f(coarse(f2c(y_n[i-1]), t[i-1], t[i])) \
       + fine(y[i-1], t[i-1], t[i]) \
       - c2f(coarse(f2c(y[i-1]), t[i-1], t[i]))
```

The rest is boiler plate. For the `c2f` and `f2c` mappings we provide a default argument of the identity function.

``` {.python title="parareal/parareal.py"}
from .abstract import (Solution, Mapping)
import numpy as np

def identity(x):
    return x

def parareal(
        coarse: Solution,
        fine: Solution,
        c2f: Mapping = identity,
        f2c: Mapping = identity):
    def f(y, t):
        m = t.size
        y_n = [None] * m
        y_n[0] = y[0]
        for i in range(1, m):
            <<parareal-core-2>>
        return y_n
    return f

def parareal_np(
        coarse: Solution,
        fine: Solution,
        c2f: Mapping = identity,
        f2c: Mapping = identity):
    def f(y, t):
        m = t.size
        y_n = np.zeros_like(y)
        y_n[0] = y[0]
        for i in range(1, m):
            <<parareal-core-2>>
        return y_n
    return f
```

## Running in parallel

``` {.python title="#import-dask"}
from dask import delayed  # type: ignore
```

``` {.python title="#daskify"}
<<import-dask>>
import numpy as np

from pintFoam.parareal.harmonic_oscillator import \
    ( harmonic_oscillator, underdamped_solution )
from pintFoam.parareal.forward_euler import \
    ( forward_euler )
from pintFoam.parareal.tabulate_solution import \
    ( tabulate )
from pintFoam.parareal.parareal import \
    ( parareal )
from pintFoam.parareal.iterate_solution import \
    ( iterate_solution)


attrs = {}

def green(f):
    def greened(*args):
        node = f(*args)
        attrs[node.key] = {"fillcolor": "#8888cc", "style": "filled"}
        return node
    return greened

@delayed
def gather(*args):
    return list(args)
```

To see what Dask does, first we'll daskify the direct integration routine in `tabulate`. We take the same harmonic oscillator we had before. For the sake of argument let's divide the time line in three steps (so four points).

``` {.python title="#daskify"}
omega_0 = 1.0
zeta = 0.5
f = harmonic_oscillator(omega_0, zeta)
t = np.linspace(0.0, 15.0, 4)
```

We now define the `fine` integrator:

```{.python title="#daskify"}
h = 0.01

@green
@delayed
def fine(x, t_0, t_1):
    return iterate_solution(forward_euler(f), h)(x, t_0, t_1)
```

It doesn't really matter what the fine integrator does, since we won't run anything. We'll just pretend. The `delayed` decorator makes sure that the integrator is never called, we just store the information that we *want* to call the `fine` function. The resulting value is a *promise* that at some point we *will* call the `fine` function. The nice thing is, that this promise behaves like any other Python object, it even qualifies as a `Vector`! The `tabulate` routine returns a `Sequence` of `Vector`s, in this case a list of promises. The `gather` function takes a list of promises and turns it into a promise of a list.

``` {.python title="#daskify"}
y_euler = tabulate(fine, [1.0, 0.0], t)
```

We can draw the resulting workflow:

``` {.python title="build/plot-dask-seq.py"}
<<daskify>>

gather(*y_euler).visualize("docs/img/seq-graph.svg", rankdir="LR", data_attributes=attrs)
```

``` {.make .figure target=img/seq-graph.svg}
Sequential integration
---
$(target): build/plot-dask-seq.py
> @mkdir -p $(@D)
> python $<
```

This workflow is entirely sequential, every step depending on the preceding one. Now for Parareal! We also define the `coarse` integrator.

``` {.python title="#daskify"}
@delayed
def coarse(x, t_0, t_1):
    return forward_euler(f)(x, t_0, t_1)
```

Parareal is initialised with the ODE integrated by the coarse integrator, just like we did before with the fine one.

``` {.python title="#daskify"}
y_first = tabulate(coarse, [1.0, 0.0], t)
```

We can now perform a single iteration of Parareal to see what the workflow looks like:

``` {.python title="#daskify"}
y_parareal = gather(*parareal(coarse, fine)(y_first, t))
```

``` {.python title="build/plot-dask-graphs.py"}
<<daskify>>

y_parareal.visualize("docs/img/parareal-graph.svg", rankdir="LR", data_attributes=attrs)
```

``` {.make .figure target=img/parareal-graph.svg}
Parareal iteration; the fine integrators (marked with blue squares) can be run in parallel.
---
$(target): build/plot-dask-graphs.py
> @mkdir -p $(@D)
> python $<
```

## Dask futures
We reimplement Parareal in the `futures` framework of Dask. We have a few helper functions: `identity` to be used as default instance for the mappings between coarse and fine meshes, and `pairs`, a function that iterates through successive pairs of a list.

``` {.python title="parareal/futures.py"}
from .abstract import (Solution, Mapping, Vector)
from typing import (Callable)
from dataclasses import dataclass
from math import ceil
import numpy as np
from numpy.typing import NDArray
from dask.distributed import Client, Future  # type: ignore
import logging


def identity(x):
    return x

def pairs(lst):
    return zip(lst[:-1], lst[1:])

<<parareal-futures>>
```

We need to send every operation to a remote worker, that includes summing the vectors from coarse and fine integrators.

``` {.python title="#parareal-futures"}
def combine(c1: Vector, f1: Vector, c2: Vector) -> Vector:
    return c1 + f1 - c2
```

``` {.python title="#time-windows"}
def time_windows(times, window_size):
    """Split the times vector in a set of time windows of a given size.

    Args:
        times:          The times vector
        window_size:    The number of steps per window (note that n steps
        correspond to n + 1 elements in the window). The last window may
        be smaller.
    """
    n_intervals = len(times) - 1
    n = int(ceil(n_intervals / window_size))
    m = window_size
    return [times[i*m:min(i*m+m+1, len(times))] for i in range(n)]
```

Every call that actually requires some of the data needs to be sent to the remote worker(s). Where we could get away before with putting everything in a closure, now it is easier to make a class that includes the Dask `Client` instance.

``` {.python title="#parareal-futures"}
@dataclass
class Parareal:
    client: Client
    coarse: Callable[[int], Solution]
    fine: Callable[[int], Solution]
    c2f: Mapping = identity
    f2c: Mapping = identity

    def _c2f(self, x: Future) -> Future:
        if self.c2f is identity:
            return x
        return self.client.submit(self.c2f, x)

    def _f2c(self, x: Future) -> Future:
        if self.f2c is identity:
            return x
        return self.client.submit(self.f2c, x)

    def _coarse(self, n_iter: int, y: Future, t0: float, t1: float) ->  Future:
        logging.debug("Coarse run: %s, %s, %s", y, t0, t1)
        return self.client.submit(self.coarse(n_iter), y, t0, t1)

    def _fine(self, n_iter: int, y: Future, t0: float, t1: float) -> Future:
        logging.debug("Fine run: %s, %s, %s", y, t0, t1)
        return self.client.submit(self.fine(n_iter), y, t0, t1)

    <<parareal-methods>>
```

The `step` method implements the core parareal algorithm.

``` {.python title="#parareal-methods"}
def step(self, n_iter: int, y_prev: list[Future], t: NDArray[np.float64]) -> list[Future]:
    m = t.size
    y_next = [None] * m
    y_next[0] = y_prev[0]

    for i in range(1, m):
        c1 = self._c2f(self._coarse(n_iter, self.f2c(y_next[i-1]), t[i-1], t[i]))
        f1 = self._fine(n_iter, y_prev[i-1], t[i-1], t[i])
        c2 = self._c2f(self._coarse(n_iter, self.f2c(y_prev[i-1]), t[i-1], t[i]))
        y_next[i] = self.client.submit(combine, c1, f1, c2)

    return y_next
```

We schedule every possible iteration of parareal as a future. The tactic is to cancel remaining jobs only once we found a converging result. This way, workers can compute next iterations, even if the last step of the previous iteration is not yet complete and tested for convergence.

``` {.python title="#parareal-methods"}
def schedule(self, y_0: Vector, t: NDArray[np.float64]) -> list[list[Future]]:
    # schedule initial coarse integration
    y_init = [self.client.scatter(y_0)]
    for (a, b) in pairs(t):
        y_init.append(self._coarse(0, y_init[-1], a, b))

    # schedule all iterations of parareal
    jobs = [y_init]
    for n_iter in range(len(t)):
        jobs.append(self.step(n_iter+1, jobs[-1], t))

    return jobs
```

The `wait` method then gathers results and returns the first iteration that satisfies `convergence_test`.

``` {.python title="#parareal-methods"}
def wait(self, jobs, convergence_test):
    for i in range(len(jobs)):
        result = self.client.gather(jobs[i])
        if convergence_test(result):
            for j in jobs[i+1:]:
                self.client.cancel(j, force=True)
            return result
    return result
```

## Harmonic oscillator
We may test this on the harmonic oscillator.

``` {.python title="test/test_futures.py"}
from dataclasses import dataclass, field
from functools import partial
import logging
from numpy.typing import NDArray
import numpy as np

from parareal.futures import Parareal
from parareal.harmonic_oscillator import harmonic_oscillator
from parareal.forward_euler import forward_euler
from parareal.iterate_solution import iterate_solution
from parareal.tabulate_solution import tabulate

from dask.distributed import Client  # type: ignore


OMEGA0 = 1.0
ZETA = 0.5
H = 0.001
system = harmonic_oscillator(OMEGA0, ZETA)


def coarse(_, y, t0, t1):
    return forward_euler(system)(y, t0, t1)


def fine(_, y, t0, t1):
    return iterate_solution(forward_euler(system), H)(y, t0, t1)


@dataclass
class History:
    history: list[NDArray[np.float64]] = field(default_factory=list)

    def convergence_test(self, y):
        self.history.append(np.array(y))
        logging.debug("got result: %s", self.history[-1])
        if len(self.history) < 2:
            return False
        return np.allclose(self.history[-1], self.history[-2], atol=1e-4)


def test_parareal():
    client = Client()
    p = Parareal(client, lambda n: partial(coarse, n), lambda n: partial(fine, n))
    t = np.linspace(0.0, 15.0, 30)
    y0 = np.array([0.0, 1.0])
    history = History()
    jobs = p.schedule(y0, t)
    p.wait(jobs, history.convergence_test)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    y0 = np.array([1.0, 0.0])
    t = np.linspace(0.0, 15.0, 30)
    result = tabulate(fine, y0, t)
    print(result)
```

