import torch
import torch.nn.functional as F
from torch import Tensor

from .scores import ABCScore


def igeoodlogits_vec(logits, temperature, centroids, epsilon=1e-12, *args, **kwargs):
    logits = torch.sqrt(F.softmax(logits / temperature, dim=1))
    centroids = torch.sqrt(F.softmax(centroids / temperature, dim=1))
    mult = logits @ centroids.T
    stack = 2 * torch.acos(torch.clamp(mult, -1 + epsilon, 1 - epsilon))
    return stack


class IgeoodLogits(ABCScore):
    def __init__(self, temperature=1.0, *args, **kwargs):
        self.centroids = None
        self.temperature = temperature

    def fit(self, X: Tensor, labels: Tensor, *args, **kwargs):
        self.unique_classes = torch.unique(labels).detach().cpu().numpy().tolist()
        self.centroids = self.calculate_centroids(X, labels)

    def calculate_centroids(self, X: Tensor, targets: Tensor, *args, **kwargs):
        centroids = {}
        for c in self.unique_classes:
            centroids[c] = torch.mean(X[targets == c], dim=0, keepdim=True)
        return torch.vstack(list(centroids.values())).to(X.device)

    def forward(self, x: Tensor, *args, **kwargs):
        stack = igeoodlogits_vec(
            x, temperature=self.temperature, centroids=self.centroids
        )
        return -torch.sum(stack, dim=1, keepdim=True)
