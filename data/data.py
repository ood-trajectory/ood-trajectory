import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from dotenv import load_dotenv

from .mos_inaturalist import MOSiNaturalist
from .mos_places365 import MOSPlaces365
from .mos_sun import MOSSUN

load_dotenv(".env")

DATASETS_DIR = os.path.expanduser(os.environ.get("DATASETS_DIR", "./data"))
IMAGENET_ROOT = os.path.expanduser(os.environ.get("IMAGENET_ROOT", DATASETS_DIR))


@dataclass
class DatasetInfo:
    name: str
    n_channels: int = 3
    n_classes: int = None
    train_statistics: tuple = None
    size: int = None
    dataset_class: torch.utils.data.Dataset = None
    kwargs: Dict[str, Any] = None

    def get(self, root, split=None, transform=None):
        if self.kwargs is not None:
            return self.dataset_class(
                root=root,
                split=split,
                transform=transform,
                download=True,
                **self.kwargs
            )
        return self.dataset_class(
            root=root, split=split, transform=transform, download=True
        )

    def train_transformation(self):
        return None

    def test_transformation(self):
        return None

    def get_train_set(self, root, transform=None, *args, **kwargs):
        return self.get(root=root, split="train", transform=transform, *args, **kwargs)

    def get_test_set(self, root, transform=None, *args, **kwargs):
        return self.get(root=root, split="test", transform=transform, *args, **kwargs)


class ILSVRC2012(DatasetInfo):
    def __init__(self):
        name = "ILSVRC2012"
        n_classes = 1000
        n_channels = 3
        train_statistics = (
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        size = 224
        dataset_class = torchvision.datasets.ImageNet
        super().__init__(
            name, n_channels, n_classes, train_statistics, size, dataset_class
        )

    def get(self, root=None, split=None, transform=None, *args, **kwargs):
        if split == "test":
            split = "val"
        imagenet = torchvision.datasets.ImageNet(
            root=IMAGENET_ROOT, split=split, transform=transform
        )
        return imagenet

    def train_transformation(self):
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*self.train_statistics),
            ]
        )

    def test_transformation(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(*self.train_statistics),
            ]
        )


class Textures(DatasetInfo):
    def __init__(self):
        name = "Textures"
        n_classes = 47
        n_channels = 3
        dataset_class = torchvision.datasets.DTD
        super().__init__(
            name,
            n_channels,
            n_classes,
            train_statistics=None,
            size=None,
            dataset_class=dataset_class,
        )

    def get(self, root=None, split=None, transform=None, *args, **kwargs):
        train_dataset = self.dataset_class(
            root=root, split="train", transform=transform, download=True
        )
        val_dataset = self.dataset_class(
            root=root, split="val", transform=transform, download=True
        )
        test_dataset = self.dataset_class(
            root=root, split="test", transform=transform, download=True
        )
        dataset = torch.utils.data.ConcatDataset(
            [train_dataset, val_dataset, test_dataset]
        )

        return dataset


class Datasets(Enum):
    ILSVRC2012 = ILSVRC2012()

    TEXTURES = Textures()
    MOS_SUN = DatasetInfo("SUN", dataset_class=MOSSUN)
    MOS_INATURALIST = DatasetInfo("iNaturalist", dataset_class=MOSiNaturalist)
    MOS_PLACES = DatasetInfo("Places", dataset_class=MOSPlaces365)

    @staticmethod
    def names():
        return list(map(lambda c: c.name, Datasets))

    @staticmethod
    def pretty_names():
        return {c.name: c.value.name for c in Datasets}


DATASETS = Datasets.names()
DATASETS_PRETTY = Datasets.pretty_names()


def get_dataset_info_by_name(dataset_name: str) -> DatasetInfo:
    dataset_name = dataset_name.upper()
    return Datasets[dataset_name].value


def get_dataset(dataset_name, transform=None, train=False, root=DATASETS_DIR):
    dataset = get_dataset_info_by_name(dataset_name)
    if train:
        return dataset.get_train_set(root, transform)
    return dataset.get_test_set(root, transform)


def pretty_name_mapper(dataset_name: str):
    try:
        return get_dataset_info_by_name(dataset_name.upper()).name
    except:
        return dataset_name.upper()
