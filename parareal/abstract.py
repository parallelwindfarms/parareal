# ~\~ language=Python filename=parareal/abstract.py
# ~\~ begin <<docs/01-dho-simple.md|parareal/abstract.py>>[init]
from __future__ import annotations
from typing import (Callable, Protocol, TypeVar, Union)

# ~\~ begin <<docs/01-dho-simple.md|abstract-types>>[init]
TVector = TypeVar("TVector", bound="Vector")

class Vector(Protocol):
    def __add__(self: TVector, other: TVector) -> TVector:
        ...

    def __sub__(self: TVector, other: TVector) -> TVector:
        ...

    def __mul__(self: TVector, other: float) -> TVector:
        ...

    def __rmul__(self: TVector, other: float) -> TVector:
        ...

# ~\~ end
# ~\~ begin <<docs/01-dho-simple.md|abstract-types>>[1]
Mapping = Callable[[TVector], TVector]
# ~\~ end
# ~\~ begin <<docs/01-dho-simple.md|abstract-types>>[2]
Problem = Callable[[TVector, float], TVector]
# ~\~ end
# ~\~ begin <<docs/01-dho-simple.md|abstract-types>>[3]
Solution = Union[Callable[[TVector, float, float], TVector],
                 Callable[..., TVector]]
# ~\~ end
# ~\~ end
