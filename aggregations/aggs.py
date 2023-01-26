from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest as IForest
from sklearn.svm import OneClassSVM


class AggregationABC(ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, X: np.ndarray, *args, **kwargs):
        ...

    @abstractmethod
    def forward(self, x: np.ndarray, *args, **kwargs):
        ...

    def __call__(self, x, *args: Any, **kwargs: Any) -> Any:
        return self.forward(x, *args, **kwargs)


class MahalanobisAgg(AggregationABC):
    def fit(self, X, *args, **kwargs):
        self.mean = np.mean(X, axis=0, keepdims=True)
        if X.shape[1] == 1:
            self.cov = np.std(X, axis=0, keepdims=True) ** 2
            self.inv = np.diag(1 / self.cov)
        else:
            self.cov = np.cov(X.T)
            self.inv_cov = np.linalg.pinv(self.cov)

    def forward(self, x, *args, **kwargs):
        return -(((x - self.mean) @ self.inv_cov) * (x - self.mean)).sum(axis=1)


class ClassCondMahalanobisAgg(AggregationABC):
    def fit(self, X, pred, *args, **kwargs):
        self.n_classes = int(np.unique(pred).max() + 1)
        self.mean = {}
        self.inv_cov = {}
        for c in range(self.n_classes):
            self.cov = np.cov(X[pred == c].T)
            self.inv_cov[c] = np.linalg.pinv(self.cov)
            self.mean[c] = np.mean(X[pred == c], axis=0, keepdims=True)

    def forward(self, x, pred, *args, **kwargs):
        scores = []
        for c in range(self.n_classes):
            scores.append(
                -(
                    ((x[pred == c] - self.mean[c]) @ self.inv_cov[c])
                    * (x[pred == c] - self.mean[c])
                ).sum(axis=1)
            )
        return np.concatenate(scores)


class EuclidesAgg(AggregationABC):
    def fit(self, X, *args, **kwargs):
        self.mean = np.mean(X, axis=0, keepdims=True)

    def forward(self, x, *args, **kwargs):
        return -np.sqrt(((x - self.mean) ** 2).sum(axis=1))


class ClassCondEuclidesAgg(AggregationABC):
    def fit(self, X, pred, *args, **kwargs):
        self.n_classes = int(np.unique(pred).max() + 1)
        self.mean = {}
        for c in range(self.n_classes):
            self.mean[c] = np.mean(X[pred == c, :], axis=0, keepdims=True)

    def forward(self, x, pred, *args, **kwargs):
        scores = []
        for c in range(self.n_classes):
            scores.append(-np.sqrt(((x[pred == c, :] - self.mean[c]) ** 2).sum(axis=1)))
        return np.concatenate(scores)


class OneClassSVMAgg(AggregationABC):
    def __init__(self, *args, **kwargs):
        self.clf = OneClassSVM(gamma="auto")

    def fit(self, X, pred=None, *args, **kwargs):
        assert isinstance(X, np.ndarray)
        # randomly select 10% of the data
        idx = np.random.choice(X.shape[0], int(X.shape[0] * 0.01), replace=False)
        self.clf.fit(X[idx, :])

    def forward(self, x, pred=None, *args, **kwargs):
        assert isinstance(x, np.ndarray)
        return self.clf.score_samples(x)


class ClassCondOneClassSVMAgg(AggregationABC):
    def __init__(self, *args, **kwargs) -> None:
        self.gamma = "auto"

    def fit(self, X, pred, *args, **kwargs):
        assert isinstance(pred, np.ndarray)
        assert isinstance(X, np.ndarray)
        self.n_classes = int(np.unique(pred).max() + 1)
        self.clf = {}
        for c in range(self.n_classes):
            self.clf[c] = OneClassSVM(gamma=self.gamma).fit(X[pred == c, :])

        assert self.clf[0] != self.clf[1]

    def forward(self, x, pred, *args, **kwargs):
        assert isinstance(pred, np.ndarray)
        assert isinstance(x, np.ndarray)
        scores = []
        for c in range(self.n_classes):
            if len(x[pred == c, :] > 0):
                scores.append(self.clf[c].score_samples(x[pred == c, :]))
        scores = np.concatenate(scores)
        assert len(scores) == len(x)
        return scores


class IsolationForest(AggregationABC):
    def __init__(self, ntrees=100, sample_size=1024) -> None:
        self.ntrees = ntrees
        self.sample_size = sample_size
        self.forest = IForest(
            n_estimators=self.ntrees, max_samples=self.sample_size, contamination=1e-6
        )

    def fit(self, X, *args, **kwargs):
        assert isinstance(X, np.ndarray)
        self.forest.fit(X)

    def forward(self, x, *args, **kwargs):
        assert isinstance(x, np.ndarray)
        return self.forest.score_samples(x)


class ClassCondIsolationForest(AggregationABC):
    def __init__(self, ntrees=100, sample_size=512) -> None:
        self.ntrees = ntrees
        self.sample_size = sample_size

    def fit(self, X, pred, *args, **kwargs):
        assert isinstance(pred, np.ndarray)
        assert isinstance(X, np.ndarray)
        self.n_classes = int(np.unique(pred).max() + 1)
        self.forest = {}
        for c in range(self.n_classes):
            self.forest[c] = IForest(
                n_estimators=self.ntrees,
                max_samples=self.sample_size,
                contamination=1e-4,
            ).fit(X[pred == c, :])

    def forward(self, x, pred, *args, **kwargs):
        assert isinstance(pred, np.ndarray)
        assert isinstance(x, np.ndarray)
        scores = []
        for c in range(self.n_classes):
            if len(x[pred == c, :] > 0):
                scores.append(self.forest[c].score_samples(x[pred == c, :]))
        scores = np.concatenate(scores)
        assert len(scores) == len(x)
        return scores


class InnerProductAgg(AggregationABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean = None

    def fit(self, X, *args, **kwargs):
        self.mean = np.mean(X, axis=0, keepdims=True)

    def forward(self, x, *args, **kwargs):
        return np.inner(x, self.mean).reshape(-1) / np.sum(self.mean**2)


class IForestAgg(AggregationABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.forest = None

    def fit(self, X, *args, **kwargs):
        self.forest = IForest(n_estimators=100, max_samples=1024, contamination=1e-6)
        self.forest.fit(X)

    def forward(self, x, *args, **kwargs):
        return self.forest.score_samples(x)


class Aggregations(Enum):

    MAHALANOBIS = MahalanobisAgg
    CLASS_MAHALANOBIS = ClassCondMahalanobisAgg
    EUCLIDES = EuclidesAgg
    CLASS_EUCLIDES = ClassCondEuclidesAgg
    ONE_CLASS_SVM = OneClassSVMAgg
    CLASS_ONE_CLASS_SVM = ClassCondOneClassSVMAgg
    IFOREST = IForestAgg

    INNERPRODUCT = InnerProductAgg


def get_aggregation(name: str, *args, **kwargs) -> AggregationABC:
    try:
        return Aggregations[name.upper()].value(*args, **kwargs)
    except:
        return Aggregations[name.upper()].value
