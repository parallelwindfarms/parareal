# ~\~ language=Python filename=parareal/harmonic_oscillator.py
# ~\~ begin <<lit/01-parareal.md|parareal/harmonic_oscillator.py>>[init]
from .abstract import (Problem)
from typing import Callable
from numpy.typing import NDArray
import numpy as np

def harmonic_oscillator(omega_0: float, zeta: float) -> Problem:
    def f(y, t):
        return np.r_[y[1], -2 * zeta * omega_0 * y[1] - omega_0**2 * y[0]]
    return f

# ~\~ begin <<lit/01-parareal.md|harmonic-oscillator-solution>>[init]
def underdamped_solution(omega_0: float, zeta: float) \
        -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    amp   = 1 / np.sqrt(1 - zeta**2)
    phase = np.arcsin(zeta)
    freq  = omega_0 * np.sqrt(1 - zeta**2)

    def f(t: NDArray[np.float64]) -> NDArray[np.float64]:
        dampening = np.exp(-omega_0*zeta*t)
        q = amp * dampening * np.cos(freq * t - phase)
        p = - amp * omega_0 * dampening * np.sin(freq * t)
        return np.c_[q, p]
    return f
# ~\~ end

if __name__ == "__main__":
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from plotnine import ggplot, geom_line, aes  # type: ignore

    from pintFoam.parareal.harmonic_oscillator import harmonic_oscillator
    from pintFoam.parareal.forward_euler import forward_euler
    from pintFoam.parareal.iterate_solution import iterate_solution
    from pintFoam.parareal.tabulate_solution import tabulate_np

    OMEGA0 = 1.0
    ZETA = 0.5
    H = 0.001
    system = harmonic_oscillator(OMEGA0, ZETA)

    def coarse(y, t0, t1):
        return forward_euler(system)(y, t0, t1)

    # fine :: Solution[NDArray]
    def fine(y, t0, t1):
        return iterate_solution(forward_euler(system), H)(y, t0, t1)

    y0 = np.array([1.0, 0.0])
    t = np.linspace(0.0, 15.0, 100)
    exact_result = underdamped_solution(OMEGA0, ZETA)(t)
    euler_result = tabulate_np(fine, y0, t)

    data = pd.DataFrame({
        "time": t,
        "exact_q": exact_result[:,0],
        "exact_p": exact_result[:,1],
        "euler_q": euler_result[:,0],
        "euler_p": euler_result[:,1]})

    plot = ggplot(data) \
        + geom_line(aes("time", "exact_q")) \
        + geom_line(aes("time", "euler_q"), color="#000088")
    plot.save("plot.svg")

# ~\~ end
