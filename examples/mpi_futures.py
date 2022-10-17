# ~\~ language=Python filename=examples/mpi_futures.py
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi>>[init]
from __future__ import annotations
import argh  # type: ignore
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import (Union, Callable, Optional, Any, Iterator)

# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi-imports>>[init]
from dask_mpi import initialize  # type: ignore
from dask.distributed import Client  # type: ignore
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi-imports>>[1]
import operator
from functools import partial
import h5py as h5  # type: ignore
from abc import (ABC, abstractmethod)
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi-imports>>[2]
from parareal.futures import (Parareal)

from parareal.forward_euler import forward_euler
# from pintFoam.parareal.iterate_solution import iterate_solution
from parareal.tabulate_solution import tabulate
from parareal.harmonic_oscillator import (underdamped_solution, harmonic_oscillator)

import math
# from uuid import uuid4
import logging
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|vector-expressions>>[init]
class Vector(ABC):
    @abstractmethod
    def reduce(self: Vector) -> np.ndarray:
        pass

    def __add__(self, other):
        return BinaryExpr(operator.add, self, other)

    def __sub__(self, other):
        return BinaryExpr(operator.sub, self, other)

    def __mul__(self, scale):
        return UnaryExpr(partial(operator.mul, scale), self)

    def __rmul__(self, scale):
        return UnaryExpr(partial(operator.mul, scale), self)
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|vector-expressions>>[1]
def reduce_expr(expr: Union[np.ndarray, Vector]) -> np.ndarray:
    while isinstance(expr, Vector):
        expr = expr.reduce()
    return expr
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|vector-expressions>>[2]
@dataclass
class H5Snap(Vector):
    path: Path
    loc: str
    slice: list[Union[None, int, slice]]

    def data(self):
        with h5.File(self.path, "r") as f:
            return f[self.loc].__getitem__(tuple(self.slice))

    def reduce(self):
        x = self.data()
        logger = logging.getLogger()
        logger.debug(f"read {x} from {self.path}")
        return self.data()
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|vector-expressions>>[3]
class Index:
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return list(idx)
        else:
            return [idx]

index = Index()
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|vector-expressions>>[4]
@dataclass
class UnaryExpr(Vector):
    func: Callable[[np.ndarray], np.ndarray]
    inp: Vector

    def reduce(self):
        a = reduce_expr(self.inp)
        return self.func(a)


@dataclass
class BinaryExpr(Vector):
    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    inp1: Vector
    inp2: Vector

    def reduce(self):
        a = reduce_expr(self.inp1)
        b = reduce_expr(self.inp2)
        return self.func(a, b)
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|vector-expressions>>[5]
@dataclass
class LiteralExpr(Vector):
    value: np.ndarray

    def reduce(self):
        return self.value
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi-coarse>>[init]
@dataclass
class Coarse:
    n_iter: int
    system: Any

    def solution(self, y, t0, t1):
        a = LiteralExpr(forward_euler(self.system)(reduce_expr(y), t0, t1))
        logging.debug(f"coarse result: {y} {reduce_expr(y)} {t0} {t1} {a}")
        return a
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi-fine>>[init]
def generate_filename(name: str, n_iter: int, t0: float, t1: float) -> str:
    return f"{name}-{n_iter:04}-{int(t0*1000):06}-{int(t1*1000):06}.h5"

@dataclass
class Fine:
    parent: Path
    name: str
    n_iter: int
    system: Any
    h: float

    def solution(self, y, t0, t1):
        logger = logging.getLogger()
        n = math.ceil((t1 - t0) / self.h)
        t = np.linspace(t0, t1, n + 1)

        self.parent.mkdir(parents=True, exist_ok=True)
        path = self.parent / generate_filename(self.name, self.n_iter, t0, t1)

        with h5.File(path, "w") as f:
            logger.debug("fine %f - %f", t0, t1)
            y0 = reduce_expr(y)
            logger.debug(":    %s -> %s", y, y0)
            x = tabulate(forward_euler(self.system), reduce_expr(y), t)
            ds = f.create_dataset("data", data=x)
            ds.attrs["t0"] = t0
            ds.attrs["t1"] = t1
            ds.attrs["h"] = self.h
            ds.attrs["n"] = n
        return H5Snap(path, "data", index[-1])
# ~\~ end
# ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi-history>>[init]
@dataclass
class History:
    archive: Path
    history: list[list[Vector]] = field(default_factory=list)

    def convergence_test(self, y) -> bool:
        logger = logging.getLogger()
        self.history.append(y)
        if len(self.history) < 2:
            return False
        a = np.array([reduce_expr(x) for x in self.history[-2]])
        b = np.array([reduce_expr(x) for x in self.history[-1]])
        maxdif = np.abs(a - b).max()
        converged = maxdif < 1e-4
        logger.info("maxdif of %f", maxdif)
        if converged:
            logger.info("Converged after %u iteration", len(self.history))
        return converged
# ~\~ end

def get_data(files: list[Path]) -> Iterator[np.ndarray]:
    for n in files:
        with h5.File(n, "r") as f:
            yield f["data"][:]

def combine_fine_data(files: list[Path]) -> np.ndarray:
    data = get_data(files)
    first = next(data)
    return np.concatenate([first] + [x[1:] for x in data], axis=0)

# def list_files(path: Path) -> list[Path]:
#     all_files = path.glob("*.h5")
#     return []

def main(log: str = "WARNING", log_file: Optional[str] = None,
         OMEGA0=1.0, ZETA=0.5, H=0.01):
    """Run model of dampened hormonic oscillator in Dask"""
    log_level = getattr(logging, log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level `{log}`")
    logging.basicConfig(level=log_level, filename=log_file)
    # ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi-main>>[init]
    initialize()
    client = Client()
    # ~\~ end
    # ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi-main>>[1]
    system = harmonic_oscillator(OMEGA0, ZETA)
    # ~\~ end
    # ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi-main>>[2]
    y0 = np.array([1.0, 0.0])
    t = np.linspace(0.0, 15.0, 20)
    archive = Path("./output/euler")
    underdamped_solution(OMEGA0, ZETA)(t)
    tabulate(Fine(archive, "fine", 0, system, H).solution, LiteralExpr(y0), t)

    # euler_files = archive.glob("*.h5")
    # ~\~ end
    # ~\~ begin <<lit/03-using-hdf5-and-mpi.md|example-mpi-main>>[3]
    archive = Path("./output/parareal")
    p = Parareal(
        client,
        lambda n: Coarse(n, system).solution,
        lambda n: Fine(archive, "fine", n, system, H).solution)
    jobs = p.schedule(LiteralExpr(y0), t)
    history = History(archive)
    p.wait(jobs, history.convergence_test)
    # ~\~ end

if __name__ == "__main__":
    argh.dispatch_command(main)
# ~\~ end
