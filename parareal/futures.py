# ~\~ language=Python filename=parareal/futures.py
# ~\~ begin <<lit/02-parafutures.md|parareal-futures>>[init]
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
# ~\~ end
# ~\~ begin <<lit/02-parafutures.md|parareal-futures>>[1]
def combine(c1: Vector, f1: Vector, c2: Vector) -> Vector:
    return c1 + f1 - c2
# ~\~ end
# ~\~ begin <<lit/02-parafutures.md|parareal-futures>>[2]
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

    # ~\~ begin <<lit/02-parafutures.md|parareal-methods>>[init]
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
    # ~\~ end
    # ~\~ begin <<lit/02-parafutures.md|parareal-methods>>[1]
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
    # ~\~ end
    # ~\~ begin <<lit/02-parafutures.md|parareal-methods>>[2]
    def wait(self, jobs, convergence_test):
        for i in range(len(jobs)):
            result = self.client.gather(jobs[i])
            if convergence_test(result):
                for j in jobs[i+1:]:
                    self.client.cancel(j, force=True)
                return result
        return result
    # ~\~ end
# ~\~ end
