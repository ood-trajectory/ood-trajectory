import torch
import torch.nn.functional as F
from torch import Tensor

from .scores import ABCScore


def msp(logits) -> Tensor:
    return torch.max(F.softmax(logits, dim=1), dim=1)[0]


class MSP(ABCScore):
    """Maxiumum softmax prediction OOD detector."""

    def __init__(self):
        super().__init__("MSP")

    def fit(self, X, *args, **kwargs):
        return

    def forward(self, x, *args, **kwargs):
        return msp(x).reshape(-1, 1)
