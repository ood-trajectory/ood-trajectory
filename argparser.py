import argparse
import json
import os

"""argument parser for model, dataset name, etc."""
parser = argparse.ArgumentParser()
# global arguments
parser.add_argument("--model", type=str.upper, help="model name")
parser.add_argument("--dataset", type=str.upper, help="dataset name")
parser.add_argument("-bs", "--batch-size", type=int, help="batch size")
parser.add_argument("--seed", type=int, help="random seed")

# save features arguments
parser.add_argument("-op", "--reduction_op", type=str.lower, help="reduction operation")
parser.add_argument("--save-root", type=str, help="root directory to save features")
parser.add_argument("--train", action="store_true", help="save train dataset features")
parser.add_argument("--num_samples", type=int, help="number of samples to save")

# save functional dataset arguments
parser.add_argument("--detector", type=str.upper, help="detector name")
parser.add_argument("-outs", "--out-dataset-names", nargs="+", help="list of ood dataset names")

# multi layer ood detection arguments
parser.add_argument("-agg", "--aggregation", type=str.lower, help="aggregation method")

# baseline arguments
parser.add_argument("-t", "--temperature", type=float, help="temperature for baselines")
parser.add_argument("-eps", "--eps", type=float, help="eps for baselines")


class Arguments:
    """enumerate argument parser arguments"""

    model: str = "DENSENET121_ILSVRC2012"
    dataset: str = None
    batch_size: int = 256
    train: bool = False
    seed: int = 1
    dont_save: bool = False
    out_dataset_names: list = None

    reduction_op: str = "adaptive-max-pool2d"
    save_root: str = os.environ.get("TENSORS_DIR", "./tensors")
    num_samples: int = 100000

    detector: str = "PROJECTION"

    aggregation: str = "innerproduct"
    dataset: str = None

    temperature: float = 1.0
    eps: float = 0.0

    def __init__(self, args: argparse.Namespace):
        """override default arguments"""
        args_dict = vars(args)
        for item, value in args_dict.items():
            if value is not None:
                setattr(self, item, value)
            else:
                setattr(self, item, getattr(self, item))
        if "vit" in self.model.lower():
            self.reduction_op = "getitem"

    def update(self, **kwargs):
        """update arguments"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return json.dumps(vars(self), indent=2)


if __name__ == "__main__":
    args = Arguments(parser.parse_args())
    print(args)
