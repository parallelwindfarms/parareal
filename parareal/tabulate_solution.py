# ~\~ language=Python filename=parareal/tabulate_solution.py
# ~\~ begin <<lit/01-parareal.md|parareal/tabulate_solution.py>>[init]
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

# ~\~ begin <<lit/01-parareal.md|tabulate-np>>[init]
def tabulate_np(step: Solution, y_0: Array, t: Array) -> Array:
    y = np.zeros(dtype=y_0.dtype, shape=(t.size,) + y_0.shape)
    y[0] = y_0
    for i in range(1, t.size):
        y[i] = step(y[i-1], t[i-1], t[i])
    return y
# ~\~ end
# ~\~ end
