import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
from torchvision import transforms
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from . import resnetv2

load_dotenv(".env")

PRE_TRAINED_DIR = os.path.expanduser(os.environ.get("PRE_TRAINED_DIR", "vision/pre_trained"))


def densenet121(num_classes: int = 1000, pre_trained: bool = True):
    return torchvision.models.densenet121(
        weights=torchvision.models.DenseNet121_Weights.DEFAULT if pre_trained else None,
        num_classes=num_classes,
    )


def vit_16_b(num_classes: int = 1000, pre_trained: bool = True):
    return torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.DEFAULT if pre_trained else None,
        num_classes=num_classes,
    )


def resnet50(num_classes: int = 1000, pre_trained: bool = True):
    return torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT if pre_trained else None,
        num_classes=num_classes,
    )


def load_model_from_state_dict(path: Path, model: torch.nn.Module, map_location=torch.device("cpu")):
    parameters = torch.load(path, map_location=map_location)
    model.load_state_dict(parameters, strict=False)
    model.to(map_location)
    model.eval()
    return model


@dataclass
class ModelInfo(ABC):
    name: str
    num_classes: int
    dataset: str
    get: Callable
    feature_nodes: list = None
    path: str = None
    model_path: str = None

    def get_model(self, path=None, *args, **kwargs):
        model = self.get(self.num_classes)
        if path is not None:
            if os.path.exists(path):
                print(f"Loading model from memory ({path})")
                model = load_model_from_state_dict(path, model, map_location="cpu")
            else:
                raise ValueError(f"File path ({path}) not found!")
        return model

    def get_penultimate_feature_node(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_all_feature_nodes(self, linear=False):
        ...

    def get_every_node(self):
        return get_graph_node_names(self.get_model())

    @abstractmethod
    def test_transformation(self):
        ...


class DensenetBC121ImageNet(ModelInfo):
    # Accuracy 74.97%
    def __init__(self):
        super().__init__(
            name=r"DenseNet-121\\(ImageNet)",
            num_classes=1000,
            dataset="ilsvrc2012",
            get=densenet121,
        )
        self.model_path = ""

    def get_model(self, path=None, pre_trained=True, *args, **kwargs):
        model = self.get(self.num_classes, pre_trained=pre_trained)
        return model

    def get_all_feature_nodes(self, linear=False):
        nodes = [
            "features.transition1.pool",
            "features.transition2.pool",
            "features.transition3.pool",
            "features.norm5",
            "flatten",
        ]
        if linear:
            nodes.append("classifier")

        return nodes

    def get_every_node(self):
        all_nodes = get_graph_node_names(self.get_model())[0]
        nodes = [n for n in all_nodes if "conv" in n or "pool" in n]
        nodes.append("classifier")
        return nodes

    def test_transformation(self):
        img_size = 224
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform_test


class Vit16BImageNet(ModelInfo):
    """torchrun --nproc_per_node=8 train.py\
    --model vit_b_16 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema

    Accuracy: 81.07% (0.81068)
    Params: 86M
    """

    def __init__(self):
        super().__init__(
            name=r"VIT16B\\(ImageNet)",
            num_classes=1000,
            dataset="ilsvrc2012",
            get=vit_16_b,
        )
        self.model_path = ""

    def get_model(self, path=None, pre_trained=True, *args, **kwargs):
        model = self.get(self.num_classes, pre_trained=pre_trained)
        return model

    def get_all_feature_nodes(self, linear=False):
        nodes = []
        for l in range(12):
            nodes.append(f"encoder.layers.encoder_layer_{l}")
        nodes.append("encoder.ln")
        nodes.append("getitem_5")
        if linear:
            nodes.append("heads.head")
        return nodes

    def test_transformation(self):
        img_size = 224
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform_test


class BITSR101(ModelInfo):
    def __init__(self):
        super().__init__(
            name=r"\multirowcell{2}{BiT-S R101\\(ImageNet)}",
            num_classes=1000,
            dataset="ilsvrc2012",
            get=None,
        )
        self.model_type = "BiT-S-R101x1"
        self.model_path = os.path.join(PRE_TRAINED_DIR, "bitsr101_ilsvrc2012", "BiT-S-R101x1.npz")
        self.download()

    def download(self):
        url = "https://storage.googleapis.com/bit_models/BiT-S-R101x1.npz"
        cached_file = self.model_path
        if not os.path.exists(cached_file):
            os.makedirs(os.path.dirname(cached_file), exist_ok=True)
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            torch.hub.download_url_to_file(url, cached_file, hash_prefix=None, progress=True)

    def get_all_feature_nodes(self, linear=False):
        nodes = ["body.block1", "body.block2", "body.block3", "body.block4"]
        if linear:
            nodes.append("head.flatten")
        return nodes

    def get_every_node(self):
        nodes = ["body.block1", "body.block2", "body.block3", "body.block4"]
        all_nodes = get_graph_node_names(self.get_model())[0]
        nodes += [n for n in all_nodes if "conv3.conv2d" in n]
        nodes.append("head.flatten")
        return nodes

    def get_model(self, path=None, pre_trained=True, *args, **kwargs):
        model = resnetv2.KNOWN_MODELS[self.model_type](head_size=self.num_classes)
        if pre_trained:
            w = np.load(self.model_path)
            model.load_from(w)
        return model.features

    def test_transformation(self):
        img_size = 480
        transform_test = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform_test


class ResNet50ImageNet(ModelInfo):
    def __init__(self):
        super().__init__(
            name=r"\multirowcell{2}{ResNet-50\\(ImageNet)}",
            num_classes=1000,
            dataset="ilsvrc2012",
            get=resnet50,
        )
        self.model_path = ""

    def get_model(self, path=None, pre_trained=True, *args, **kwargs):
        model = self.get(self.num_classes, pre_trained=pre_trained)
        return model

    def get_all_feature_nodes(self, linear=True):
        nodes = ["layer1", "layer2", "layer3", "layer4", "flatten"]
        if linear:
            nodes.append("fc")

        return nodes

    def get_every_node(self):
        all_nodes = get_graph_node_names(self.get_model())[0]
        nodes = [n for n in all_nodes if "conv" in n or "pool" in n]
        nodes.append("classifier")
        return nodes

    def test_transformation(self):
        img_size = 224
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform_test


class Models(Enum):
    RESNET50_ILSVRC2012 = ResNet50ImageNet()
    DENSENET121_ILSVRC2012 = DensenetBC121ImageNet()
    VIT16B_ILSVRC2012 = Vit16BImageNet()
    BITSR101_ILSVRC2012 = BITSR101()

    @staticmethod
    def names():
        return list(map(lambda c: c.name, Models))

    @staticmethod
    def pretty_names():
        return {c.name: c.value.name if hasattr(c.value, "name") else "" for c in Models}


MODEL_NAMES = Models.names()
MODEL_NAMES_PRETTY = Models.pretty_names()


def _get_model_enum(model_name: str):
    return Models[model_name.upper()]


def get_model_name(model_name: str) -> ModelInfo:
    return _get_model_enum(model_name).name


def get_model_info(model_name: str) -> ModelInfo:
    return _get_model_enum(model_name).value


def get_in_dataset_name(model_name: str) -> str:
    return get_model_info(model_name).dataset.upper()


def get_num_classes(model_name: str):
    return get_model_info(model_name.upper()).num_classes


def get_pre_trained_model_by_name(
    model_name: str, root: str = PRE_TRAINED_DIR, pre_trained=True, seed=1, path=None
) -> torch.nn.Module:
    model_name = get_model_name(model_name)
    if path is None:
        path = os.path.join(root, model_name.upper() + "_" + str(seed) + ".pt")
    model_info = get_model_info(model_name)
    return model_info.get_model(path, pre_trained=pre_trained)


def get_model_features_nodes_by_name(nn_name, linear_nodes, every_node=False):
    model_info = get_model_info(nn_name)
    if every_node:
        return model_info.get_every_node()
    nodes = model_info.get_all_feature_nodes(linear_nodes)
    return nodes


def get_feature_extractor_by_name(nn_name: str, linear_nodes=True, pre_trained=True, seed=1) -> torch.nn.Module:
    model_info = get_model_info(nn_name)
    model_name = get_model_name(nn_name)

    if pre_trained:
        model = get_pre_trained_model_by_name(model_name, seed=seed)
    else:
        model = model_info.get_model(None)

    nodes = get_model_features_nodes_by_name(nn_name, linear_nodes)
    return create_feature_extractor(model, return_nodes=nodes)


def get_data_transform(nn_name: str, train: bool = False):
    model_info = get_model_info(nn_name)
    if train:
        return model_info.train_transformation()
    return model_info.test_transformation()


def get_model_path(model_name: str):
    model_info = get_model_info(model_name)
    return model_info.model_path


if __name__ == "__main__":
    # print(get_graph_node_names(densenet121())[0])
    # print(get_graph_node_names(vit_16_b())[0])
    # print(get_graph_node_names(BITSR101().get_model())[0])

    model = get_feature_extractor_by_name("BITSR101_ILSVRC2012", True, True)
    x = torch.randn(1, 3, 480, 480)
    feats = model(x)
    for i, (k, v) in enumerate(feats.items()):
        print(i, k, type(v))
