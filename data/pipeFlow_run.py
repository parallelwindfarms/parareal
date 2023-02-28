from pathlib import Path
from collections.abc import Sequence
import numpy as np
from math import ceil
import uuid

from dask import delayed
from functools import partial
from pintFoam.parareal import (parareal, tabulate)
from pintFoam import (BaseCase, foam, block_mesh)
from pintFoam.vector import (Vector)
from pintFoam.foam import (map_fields)
from pintFoam.utils import (generate_job_name)

fields = ["p", "U"]
case_name = "pipeFlow"

fine_case = BaseCase(Path(case_name + "Fine"), "baseCase", fields=fields)
coarse_case = BaseCase(Path(case_name + "Coarse"), "baseCase", fields=fields)

fine_case.clean()
coarse_case.clean()

block_mesh(fine_case)
block_mesh(coarse_case)

times = np.linspace(0, 2, 21)

@delayed
def gather(*args):
    return list(args)


@delayed
def c2f(x):
    """Coarse to fine.

    Interpolate the underlying field x from the coarse to the fine grid"""
    return map_fields(x, fine_case, map_method="interpolate")


@delayed
def f2c(x):
    """Fine to coarse.

    Interpolate the underlying field x from the fine to the coarse grid"""
    return map_fields(x, coarse_case, map_method="interpolate")

@delayed
def fine(n, x, t_0, t_1):
    """Fine integrator."""
    uid = uuid.uuid4()
    return foam("pimpleFoam", 0.001, x, t_0, t_1,
                job_name=generate_job_name(n, t_0, t_1, uid, "fine"))


@delayed
def coarse(n, x, t_0, t_1):
    """Coarse integrator."""
    uid = uuid.uuid4()
    return foam("pimpleFoam", 0.1, x, t_0, t_1,
                job_name=generate_job_name(n, t_0, t_1, uid, "coarse"))


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


def solve(init: Vector, times: Sequence[float], max_iter=1) -> list[Vector]:
    # coarse initial integration from fine initial condition
    y = list(map(c2f, tabulate(partial(coarse, 0), f2c(init), times)))
    for n in range(1, max_iter+1):
        y = gather(*parareal(partial(coarse, n), partial(fine, n), c2f, f2c)(y, times))
    return y  # .compute()


def windowed(times, init, window_size):
    windows = time_windows(times, window_size)
    result = [init]
    for w in windows:
        w_result = solve(result[-1], w)
        result.extend(w_result)
    return result


# print(time_windows(np.arange(40), 11))
windows = time_windows(times, 10)
init = fine_case.new_vector()
wf = solve(init, windows[0], 3)
# wf.visualize("parareal.png")
wf.compute(n_workers=4)

