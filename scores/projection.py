import torch
import torch.nn.functional as F

from .scores import ABCScore


def cosine_layer_score(x: torch.Tensor, mus: torch.Tensor):
    stack = torch.zeros(x.shape[0], len(mus), device=x.device, dtype=x.dtype)
    for i, mu in enumerate(mus):
        stack[:, i] = F.cosine_similarity(x, mu.unsqueeze(0), dim=-1)
    return stack


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
        return torch.norm(x, p=2, dim=-1, keepdim=True) * cosine_layer_score(
            x, self.mus.to(x.device)
        )
