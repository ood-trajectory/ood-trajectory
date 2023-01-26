import json
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import data as data_utils
import nn as model_utils
from argparser import Arguments, parser

logger = logging.getLogger(__name__)


def adaptive_avg_pool2d(data: torch.Tensor, *args, **kwargs):
    return torch.flatten(torch.nn.AdaptiveAvgPool2d((1, 1))(data), 1)


def adaptive_max_pool2d(data: torch.Tensor, *args, **kwargs):
    return torch.flatten(torch.nn.AdaptiveMaxPool2d((1, 1))(data), 1)


def getitem(data: torch.Tensor, *args, **kwargs):
    return data[:, 0].clone().contiguous()


reduction_dispatcher = {
    "adaptive-avg-pool2d": adaptive_avg_pool2d,
    "adaptive-max-pool2d": adaptive_max_pool2d,
    "getitem": getitem,
}


def make_reduction(reduction_type: str, *args, **kwargs):
    def reduction(data: torch.Tensor) -> torch.Tensor:
        return reduction_dispatcher[reduction_type](data, *args, **kwargs)

    return reduction


def main(args: Arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    transform = model_utils.get_model_info(args.model).test_transformation()
    print(transform)
    dataset = data_utils.get_dataset(args.dataset, transform=transform, train=args.train)
    random_idx = np.random.choice(len(dataset), args.num_samples, replace=False)
    dataset = torch.utils.data.Subset(dataset, random_idx)
    logger.info("Sample size %s", next(iter(dataset))[0].shape)
    dataset_size = len(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
        drop_last=False,
    )

    # load model
    model = model_utils.get_feature_extractor_by_name(args.model, pre_trained=True, linear_nodes=True, seed=args.seed)
    logger.info(
        "Number of parameters in %s: %s",
        args.model,
        sum([p.numel() for p in model.parameters()]),
    )
    model = model.to(device)

    start = time.time()
    targets_list = []
    features = defaultdict(list)
    _reduction_op = make_reduction(args.reduction_op)
    with torch.no_grad():
        model.eval()
        for x, y in tqdm(data_loader):
            targets_list.append(y.to("cpu", dtype=torch.float32))
            x = x.to(device, non_blocking=True)
            feats = model(x)
            for k, v in feats.items():
                if len(v.shape) > 2:
                    v = _reduction_op(v)

                features[k].append(v.to("cpu", dtype=torch.float32))

    stop = time.time()
    logger.info("Time elapsed %s", stop - start)

    targets = torch.cat(targets_list, dim=0).numpy()
    logger.info(f"Targets shape: {targets.shape}")
    assert targets.shape[0] == dataset_size

    destination_path = os.path.join(args.save_root, args.model, args.dataset)
    os.makedirs(destination_path, exist_ok=True)
    for k in tqdm(features, "Concatenating features"):
        os.makedirs(os.path.join(destination_path, k), exist_ok=True)
        features[k] = torch.cat(features[k], dim=0).numpy()

        logger.info("Saving feature {} to {}/{}".format(k, destination_path, k))
        filename = f"features_{args.reduction_op}.npy"
        if args.train:
            filename = "train_" + filename
        np.save(os.path.join(destination_path, k, filename), features[k])

        ## accuracy
        if k == list(features.keys())[-1]:
            logger.info("Calculating accuracy")
            logits = features[k]
            accuracy = np.mean(targets.reshape(-1) == np.argmax(logits, 1).reshape(-1))
            print(logits.shape)
            filename = "logits.npy"
            if args.train:
                filename = "train_" + filename
            logger.info(f"Accuracy: {accuracy}")
            np.save(os.path.join(destination_path, filename), logits)

        features[k] = None

    # save targets
    logger.info("Saving targets to {}".format(destination_path))
    filename = "targets.npy"
    if args.train:
        filename = "train_" + filename
    np.save(os.path.join(destination_path, filename), targets)
    logger.info("Targets saved")

    # save metadata
    ks = list(features.keys())
    metadata = {"features_names": ks}
    filename = "metadata.json"
    with open(os.path.join(args.save_root, args.model, filename), "w") as fp:
        json.dump(metadata, fp)


if __name__ == "__main__":
    args = Arguments(parser.parse_args())
    if "vit" in args.model.lower():
        args.reduction_op = "getitem"
    print(args)
    logging.basicConfig(
        format="--> %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main(args)
