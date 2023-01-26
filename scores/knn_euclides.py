import torch
from torch import Tensor

from .scores import ABCScore


class KnnEuclidesScore(ABCScore):
    def __init__(self, alpha: float = 0.1, k: int = 100):
        super().__init__("Mahalanobis")
        self.alpha = alpha
        self.k = k
        self.X = None

        assert 0 < self.alpha <= 1, "alpha must be in (0, 1]"

    def fit(self, X: Tensor, *args, **kwargs):
        # random choice of alpha percent of X
        X = X[torch.randperm(X.shape[0])[: int(self.alpha * X.shape[0])]]
        self.X = X / X.norm(dim=-1, keepdim=True)

    def forward(self, x: Tensor, *args, **kwargs):
        # normalize test features
        x = x / torch.norm(x, p=2, dim=-1, keepdim=True)

        # pairwise euclidean distance between x and X
        dist = torch.cdist(x, self.X)

        # take smallest k distance for each test sample
        topk = torch.topk(dist, k=self.k, dim=-1, largest=False).values

        # return mean of top k distances
        return -topk.mean(-1)

    def __call__(self, x: Tensor, *args, **kwargs):
        return self.forward(x, *args, **kwargs)
