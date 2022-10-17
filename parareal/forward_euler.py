# ~\~ language=Python filename=parareal/forward_euler.py
# ~\~ begin <<lit/01-parareal.md|parareal/forward_euler.py>>[init]
from .abstract import (Vector, Problem, Solution)

def forward_euler(f: Problem) -> Solution:
    """Forward-Euler solver."""
    def step(y: Vector, t_0: float, t_1: float) -> Vector:
        """Stepping function of Euler method."""
        return y + (t_1 - t_0) * f(y, t_0)
    return step
# ~\~ end
