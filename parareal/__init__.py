# ~\~ language=Python filename=parareal/__init__.py
# ~\~ begin <<lit/01-parareal.md|parareal/__init__.py>>[init]
import subprocess

proc_cat = subprocess.run(
    ["cat", "README.md", "lit/03-using-hdf5-and-mpi.md", "lit/01-parareal.md",
     "lit/02-parafutures.md", "lit/end-api.md"],
    capture_output=True)
proc_eval = subprocess.run(
    ["awk", "-f", "eval_shell_pass.awk"],
    input=proc_cat.stdout, capture_output=True)
proc_label = subprocess.run(
    ["awk", "-f", "noweb_label_pass.awk"],
    input=proc_eval.stdout, capture_output=True)
__doc__ = proc_label.stdout.decode()

from .tabulate_solution import tabulate
from .parareal import parareal
from . import abstract

__all__ = ["tabulate", "parareal", "schedule", "abstract"]
# ~\~ end
