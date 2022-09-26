import torch

from .scores import ABCScore


def torch_reduction_matrix(sigma: torch.Tensor, method="pseudo"):

    if method == "cholesky":
        C = torch.linalg.cholesky(sigma)
        return torch.linalg.inv(C.T)
    elif method == "SVD":
        u, s, _ = torch.linalg.svd(sigma)
        return u @ torch.diag(torch.sqrt(1 / s))
    elif method == "pseudo":
        return torch.linalg.pinv(sigma)


def class_cond_mus_cov_inv_matrix(
    x: torch.Tensor, targets: torch.Tensor, inv_method="pseudo"
):
    unique_classes = torch.unique(targets).detach().cpu().numpy().tolist()
    class_cond_dot = {}
    class_cond_mean = {}
    for c in unique_classes:
        filt = targets == c
        temp = x[filt]
        class_cond_dot[c] = torch.cov(temp.T)
        class_cond_mean[c] = temp.mean(0, keepdim=True)
    cov_mat = sum(list(class_cond_dot.values())) / x.shape[0]
    inv_mat = torch_reduction_matrix(cov_mat, method=inv_method)
    mus = torch.vstack(list(class_cond_mean.values()))
    return mus, cov_mat, inv_mat


def mahalanobis_distance_inv(x: torch.Tensor, y: torch.Tensor, inverse: torch.Tensor):
    return torch.nan_to_num(torch.sqrt(((x - y).T * (inverse @ (x - y).T)).sum(0)), 1e9)


def mahalanobis_inv_layer_score(x: torch.Tensor, mus: torch.Tensor, inv: torch.Tensor):
    stack = torch.zeros(
        (x.shape[0], mus.shape[0]), device=x.device, dtype=torch.float32
    )
    for i, mu in enumerate(mus):
        stack[:, i] = mahalanobis_distance_inv(x, mu.reshape(1, -1), inv).reshape(-1)

    return stack.min(1, keepdim=True)[0]


class Mahalanobis(ABCScore):
    """Implement detector with mahalanobis distance."""

    def __init__(self):
        super().__init__("Mahalanobis")
        self.mus = None
        self.inv = None
        self.invs = None

    def fit(self, X: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        self.mus, _, self.inv = class_cond_mus_cov_inv_matrix(
            X, labels, inv_method="pseudo"
        )
        self.mus = self.mus.cpu()
        self.inv = self.inv.cpu()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return mahalanobis_inv_layer_score(
            x, self.mus.to(x.device), self.inv.to(x.device)
        )
