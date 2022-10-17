# ~\~ language=Python filename=parareal/parareal.py
# ~\~ begin <<lit/01-parareal.md|parareal/parareal.py>>[init]
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
            # ~\~ begin <<lit/01-parareal.md|parareal-core-2>>[init]
            y_n[i] = c2f(coarse(f2c(y_n[i-1]), t[i-1], t[i])) \
                   + fine(y[i-1], t[i-1], t[i]) \
                   - c2f(coarse(f2c(y[i-1]), t[i-1], t[i]))
            # ~\~ end
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
            # ~\~ begin <<lit/01-parareal.md|parareal-core-2>>[init]
            y_n[i] = c2f(coarse(f2c(y_n[i-1]), t[i-1], t[i])) \
                   + fine(y[i-1], t[i-1], t[i]) \
                   - c2f(coarse(f2c(y[i-1]), t[i-1], t[i]))
            # ~\~ end
        return y_n
    return f
# ~\~ end
