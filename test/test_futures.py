# ~\~ language=Python filename=test/test_futures.py
# ~\~ begin <<lit/02-parafutures.md|test/test_futures.py>>[init]
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
# ~\~ end
