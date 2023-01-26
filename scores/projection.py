import torch
from torch import Tensor

from .scores import ABCScore


def projection_layer_score(x: Tensor, mus: Tensor, eps=1e-7):
    # efficient cosine similarity
    num = x @ mus.T
    den = torch.norm(x, dim=-1, keepdim=True) @ torch.norm(mus, dim=-1, keepdim=True).T
    stack = num / (den + eps)
    # multiply by norm
    return torch.norm(x, p=2, dim=-1, keepdim=True) * stack


class Projection(ABCScore):
    def __init__(self):
        self.name = "Projection"
        self.mus = None

    def fit(self, X: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        self.mus = []
        unique_classes = torch.unique(labels).detach().cpu().numpy().tolist()
        for c in unique_classes:
            filt = labels == c
            if filt.sum() == 0:
                continue
            self.mus.append(X[filt].mean(0, keepdim=True))
        self.mus = torch.vstack(self.mus)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return torch.norm(x, p=2, dim=-1, keepdim=True) * projection_layer_score(x, self.mus.to(x.device))
