from copy import deepcopy
from enum import Enum

from ._abc import ABCScore
from .mahalanobis import Mahalanobis
from .msp import MSP
from .projection import Projection


class Scores(Enum):
    MSP = MSP()
    MAHALANOBIS = Mahalanobis()
    PROJECTION = Projection()

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
