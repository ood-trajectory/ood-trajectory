from copy import deepcopy
from enum import Enum

import torch

from ._abc import ABCScore
from .igeood import IgeoodLogits
from .knn_euclides import KnnEuclidesScore
from .mahalanobis import Mahalanobis
from .msp import MSP
from .projection import Projection


class Scores(Enum):
    MSP = MSP()
    MAHALANOBIS = Mahalanobis()
    PROJECTION = Projection()
    KNN = KnnEuclidesScore()
    IGEOOD_LOGITS = IgeoodLogits()

    @staticmethod
    def names():
        return list(map(lambda c: c.name, Scores))

    @staticmethod
    def pretty_names():
        return {
            c.name: c.value.name if hasattr(c.value, "name") else c.name for c in Scores
        }


LITE_SCORES_NAMES = Scores.names()
SCORES_NAMES_PRETTY = Scores.pretty_names()


def get_score_by_name(detector_name: str, *args, **kwargs) -> ABCScore:
    return deepcopy(Scores[detector_name.upper()].value)


def test():
    x = torch.randn(10_000, 10)
    X = torch.randn(50_000, 10)
    score = KnnEuclidesScore()
    score.fit(X)
    print(score(x))

    assert score(x).shape == (10_000,)


if __name__ == "__main__":
    test()
