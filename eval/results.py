import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

TENSORS_DIR = os.environ.get("TENSORS_DIR", "tensors")


@dataclass
class Config:
    nn_name: str = None
    dataset_name: str = None
    out_dataset: str = None
    score_method: str = None
    aggregation: str = None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return str(vars(self))


def append_results_to_file(results, filename="results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results = {k: [v] for k, v in results.items()}
    results = pd.DataFrame.from_dict(results, orient="columns")

    if not os.path.isfile(filename):
        results.to_csv(filename, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(filename, mode="a", header=False, index=False)


def get_in_tensors_path(config: Config):
    path = os.path.join(TENSORS_DIR, config.nn_name, config.dataset_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_out_tensors_path(config: Config):
    path = os.path.join(TENSORS_DIR, config.nn_name, config.out_dataset)
    os.makedirs(path, exist_ok=True)
    return path


def get_metadata_path(config: Config):
    path = os.path.join(TENSORS_DIR, config.nn_name)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "metadata.json")


def get_metadata(config: Config):
    path = get_metadata_path(config)
    with open(path, "r") as f:
        metadata = json.load(f)
    return metadata


def np_load_train_targets(config: Config) -> np.ndarray:
    name = "train_targets"
    root = get_in_tensors_path(config)
    path = os.path.join(root, name + ".npy")
    arr = np.load(path)
    return arr


def np_load_train_features(config: Config, feature_name: str, reduction_op: str):
    name = f"train_features_{reduction_op}"
    path = os.path.join(get_in_tensors_path(config), feature_name, name + ".npy")
    arr = np.load(path)
    return arr


def np_load_train_logits(config: Config) -> np.ndarray:
    name = "train_logits"
    path = os.path.join(get_in_tensors_path(config), name + ".npy")
    arr = np.load(path)
    return arr


def np_load_test_features(config: Config, feature_name: str, reduction_op: str):
    name = f"features_{reduction_op}"
    path = os.path.join(get_in_tensors_path(config), feature_name, name + ".npy")
    arr = np.load(path)
    return arr


def np_load_test_logits(config: Config):
    name = "logits"
    path = os.path.join(get_in_tensors_path(config), name + ".npy")
    arr = np.load(path)
    return arr


def np_load_out_features(config: Config, feature_name: str, reduction_op: str):
    name = f"features_{reduction_op}"
    path = os.path.join(get_out_tensors_path(config), feature_name, name + ".npy")
    arr = np.load(path)
    return arr


def np_load_out_logits(config: Config):
    name = "logits"
    path = os.path.join(get_out_tensors_path(config), name + ".npy")
    arr = np.load(path)
    return arr


def np_save_train_timeseries(config: Config, arr: np.ndarray, reduction_op: str):
    name = f"train_trajectories_{reduction_op}"
    path = os.path.join(get_in_tensors_path(config), config.score_method, name + ".npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def np_load_train_timeseries(config: Config, reduction_op: str):
    name = f"train_trajectories_{reduction_op}"
    path = os.path.join(get_in_tensors_path(config), config.score_method, name + ".npy")
    arr = np.load(path)
    return arr


def np_save_test_timeseries(config: Config, arr: np.ndarray, reduction_op: str):
    name = f"trajectories_{reduction_op}"
    path = os.path.join(get_in_tensors_path(config), config.score_method, name + ".npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def np_load_test_timeseries(config: Config, reduction_op: str):
    name = f"trajectories_{reduction_op}"
    path = os.path.join(get_in_tensors_path(config), config.score_method, name + ".npy")
    arr = np.load(path)
    return arr


def np_save_out_timeseries(config: Config, arr: np.ndarray, reduction_op: str):
    name = f"trajectories_{reduction_op}"
    path = os.path.join(get_out_tensors_path(config), config.score_method, name + ".npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def np_load_out_timeseries(config: Config, reduction_op: str):
    name = f"trajectories_{reduction_op}"
    path = os.path.join(get_out_tensors_path(config), config.score_method, name + ".npy")
    arr = np.load(path)
    return arr


def np_save_in_scores(config: Config, arr: np.ndarray, reduction_op: str):
    name = f"in_scores_{reduction_op}"
    path = os.path.join(
        get_in_tensors_path(config),
        config.score_method,
        config.aggregation,
        name + ".npy",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def np_save_out_scores(config: Config, arr: np.ndarray, reduction_op: str):
    name = f"out_scores_{reduction_op}"
    path = os.path.join(
        get_out_tensors_path(config),
        config.score_method,
        config.aggregation,
        name + ".npy",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def np_load_in_scores(config: Config, reduction_op: str):
    name = f"in_scores_{reduction_op}"
    path = os.path.join(
        get_in_tensors_path(config),
        config.score_method,
        config.aggregation,
        name + ".npy",
    )
    arr = np.load(path)
    return arr


def np_load_out_scores(config: Config, reduction_op: str):
    name = f"out_scores_{reduction_op}"
    path = os.path.join(
        get_out_tensors_path(config),
        config.score_method,
        config.aggregation,
        name + ".npy",
    )
    arr = np.load(path)
    return arr
