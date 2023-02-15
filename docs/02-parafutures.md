# Implementation of Parareal using Futures
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

