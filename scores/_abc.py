from abc import ABC, abstractmethod

from torch import Tensor


class ABCScore(ABC):
    """
    Lite abstract class for OOD detectors.
    """

    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def fit(self, X: Tensor, *args, **kwargs) -> None:
        return

    @abstractmethod
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return
