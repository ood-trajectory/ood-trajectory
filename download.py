import logging

import torch

import data as data_utils
import nn as nn_utils

logger = logging.getLogger(__name__)


def main():
    logger.warning(
        "You should download the ImageNet (ILSVRC2012) dataset manually and place it in the $IMAGENET_ROOT folder defined in the .env file"
    )
    # download datasets
    print("Downloading datasets")
    datasets = ["TEXTURES", "MOS_PLACES", "MOS_SUN", "MOS_INATURALIST"]
    for dataset in datasets:
        print(dataset)
        dataset = data_utils.get_dataset(dataset)
        logger.info("Downloaded %s", dataset)
        logger.info("Dataset size %s", len(dataset))

        img, label = next(iter(dataset))
        logger.info("image %s", img)
        logger.info("label %s", label)

    # download models
    print("Downloading models")
    models = ["DENSENET121", "VIT16B", "BITSR101", "RESNET50"]
    for model_prefix in models:
        print(model_prefix)
        model_name = model_prefix + "_ILSVRC2012"
        model = nn_utils.get_pre_trained_model_by_name(model_name)
        logger.info("Downloaded %s", model_name)
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        logger.info("Output shape %s", y.shape)
        assert y.shape == (1, 1000)

    print("Done")


if __name__ == "__main__":
    logging.basicConfig(
        format="--> %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
