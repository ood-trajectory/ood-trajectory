import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm

import eval.results as res
from argparser import Arguments, parser

logger = logging.getLogger(__name__)

plt.style.use(["science", "light", "no-latex"])
MARKERS = [".", "v", "^", "s", "P", "*", "X"] * 3
LINESTYLES = ["-", "--", "-.", ":"] * 3
COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3
IN_COLOR = COLOR_CYCLE[0]
OUT_COLOR = COLOR_CYCLE[1]
TRAIN_COLOR = COLOR_CYCLE[2]
A_COLOR = COLOR_CYCLE[3]
B_COLOR = COLOR_CYCLE[4]
C_COLOR = COLOR_CYCLE[5]
D_COLOR = COLOR_CYCLE[6]
E_COLOR = COLOR_CYCLE[7]
F_COLOR = COLOR_CYCLE[8]

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def sampled_sphere(K, d):
    np.random.seed(0)
    mean = np.zeros(d)
    identity = np.identity(d)
    U = np.random.multivariate_normal(mean=mean, cov=identity, size=K)
    return normalize(U)


def halfspace_mass(X, psi=100, lamb=0.5, X_test=None, U=None, K=1000):
    n, d = X.shape
    Score = np.zeros(n)

    mass_left = np.zeros(K)
    mass_right = np.zeros(K)
    s = np.zeros(K)

    if U is None:
        U = sampled_sphere(K, d)
    M = X @ U.T

    for i in range(K):
        try:
            subsample = np.random.choice(np.arange(n), size=psi, replace=False)
        except:
            subsample = np.random.choice(np.arange(n), size=n - 1, replace=False)
        SP = M[subsample, i]
        max_i = np.max(SP)
        min_i = np.min(SP)
        mid_i = (max_i + min_i) / 2
        s[i] = (
            lamb * (max_i - min_i) * np.random.uniform()
            + mid_i
            - lamb * (max_i - min_i) / 2
        )
        mass_left[i] = (SP < s[i]).sum() / psi
        mass_right[i] = (SP > s[i]).sum() / psi
        Score += mass_left[i] * (M[:, i] < s[i]) + mass_right[i] * (M[:, i] > s[i])

    if X_test is None:
        return Score / K
    else:
        Score_test = np.zeros(len(X_test))
        M_test = X_test @ U.T
        for i in range(K):
            Score_test += mass_left[i] * (M_test[:, i] < s[i]) + mass_right[i] * (
                M_test[:, i] > s[i]
            )
        return Score_test / K


def main():
    # load training features
    config = res.Config(nn_name=args.model, dataset_name="ILSVRC2012")
    x = torch.from_numpy(
        res.np_load_train_features(config, args.feature, args.reduction_op)
    )
    train_targets = torch.from_numpy(res.np_load_train_targets(config)).reshape(-1)
    train_logits = torch.from_numpy(res.np_load_train_logits(config))
    n_classes = train_logits.shape[1]

    features_per_class = {c: x[train_targets == c] for c in range(n_classes)}

    max_depths = []
    mean_depths = []
    limit = 1
    for j, (k, x) in enumerate(tqdm(features_per_class.items(), total=limit)):

        d = x.shape[1]
        mean = x.mean(dim=0)
        median = torch.median(x, dim=0).values
        half_space = torch.zeros(d)
        for i in range(d):
            half_space[i] = torch.median(x[x[:, i] > 0, i], dim=0).values
        mean_shift = torch.zeros(d)
        for i in range(d):
            mean_shift[i] = torch.median(x[x[:, i] > mean[i], i], dim=0).values

        # plot histograms per dimension with the mean and median
        d_max = 20
        c = 5
        fig, axs = plt.subplots(d_max // c, c, figsize=(16, d_max // c * 2.5), dpi=300)
        for i in range(d_max):
            axs[i // c, i % c].hist(x[:, i], bins=30)
            axs[i // c, i % c].axvline(
                mean[i], color="r", linestyle="dashed", linewidth=1.5
            )
            axs[i // c, i % c].axvline(
                median[i], color="g", linestyle="dashed", linewidth=1.5
            )
            axs[i // c, i % c].set_title(f"dim {i+1}")
            axs[i // c, i % c].set_xticks([])
            axs[i // c, i % c].set_yticks([])
        plt.tight_layout()
        os.makedirs("images/mean_median/", exist_ok=True)
        plt.savefig(f"images/mean_median/{k}.pdf", bbox_inches="tight", dpi=300)
        plt.close()

        max_depths.append(halfspace_mass(x.numpy(), X_test=x.numpy(), lamb=1).max())
        mean_depths.append(
            halfspace_mass(x.numpy(), X_test=mean.reshape(1, -1).numpy(), lamb=1)
        )

        if j >= limit:
            break

    # fit a pca in the data
    pca = PCA(n_components=2)
    pca.fit(x)
    x_pca = pca.transform(x)

    # plot the data in the pca space
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    ax.scatter(x_pca[:, 0], x_pca[:, 1], s=1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"images/mean_median/pca.png")
    plt.close()

    # compute the data depth of the mean on x
    mean = x.mean(dim=0)

    # depth histogram
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    ax.hist(
        np.array(max_depths),
        bins=10,
        density=True,
        label="Max depth of data points",
    )
    ax.hist(
        np.array(mean_depths),
        bins=10,
        density=True,
        label="Depth for the mean vectors",
        color=C_COLOR,
    )
    ax.set_xlabel("Depth")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"images/mean_median/depth.pdf", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser.add_argument("--feature", type=str, default="flatten")
    args = Arguments(parser.parse_args())
    logging.basicConfig(
        format="--> %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
