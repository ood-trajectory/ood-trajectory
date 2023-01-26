import logging
import os

import torch
import torch.utils.data
from torch import Tensor
from tqdm import tqdm

import eval.metrics as m
import eval.results as res
import scores
from argparser import Arguments, parser

logger = logging.getLogger(__name__)


def probability_weight_scores(scores: Tensor, probs: Tensor = None):
    if len(scores.shape) == 1:
        return scores.reshape(-1, 1)

    return torch.sum(scores * probs, dim=1, keepdim=True)


def main(args: Arguments):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", DEVICE)

    in_dataset_name = args.model.split("_")[1].upper()
    # config
    config = res.Config(nn_name=args.model, dataset_name=in_dataset_name, score_method=args.detector)
    metadata = res.get_metadata(config)
    features_names = metadata["features_names"]

    # load features
    train_targets = torch.from_numpy(res.np_load_train_targets(config)).reshape(-1)
    train_logits = torch.from_numpy(res.np_load_train_logits(config))
    train_pred = torch.argmax(train_logits, dim=1)
    n_classes = train_logits.shape[-1]
    label = train_pred
    probs = torch.softmax(train_logits, dim=1)

    test_logits = torch.from_numpy(res.np_load_test_logits(config))
    test_pred = torch.argmax(test_logits, dim=1)
    test_probs = torch.softmax(test_logits, dim=1)

    # variables to store results
    scorer = {}
    train_series = torch.zeros((len(train_logits), len(features_names)))
    in_series = torch.zeros((len(test_logits), len(features_names)))
    out_series = {name: torch.zeros((10000, len(features_names))) for name in args.out_dataset_names}
    with torch.no_grad():
        for i, f in enumerate(tqdm(features_names)):
            scorer[f] = scores.get_score_by_name(config.score_method)
            train_feature = torch.from_numpy(res.np_load_train_features(config, f, args.reduction_op))
            logger.info("Train feature shape: %s", train_feature.shape)
            test_feature = torch.Tensor(res.np_load_test_features(config, f, args.reduction_op))
            batch_size = args.batch_size
            logger.info("Fitting scorer %s", f)
            assert len(train_feature) == len(label)

            scorer[f].fit(train_feature.to(DEVICE), labels=label)
            score_fn = scorer[f].forward

            logger.info("Calculating train scores for %s", f)
            for b in tqdm(range(0, len(train_feature), batch_size), desc="Train dataset"):
                train_series[b : b + batch_size, i] = (
                    probability_weight_scores(
                        score_fn(train_feature[b : b + batch_size].to(DEVICE)),
                        pred=label[b : b + batch_size],
                        probs=probs[b : b + batch_size].to(DEVICE),
                    )
                    .reshape(-1)
                    .cpu()
                )

            logger.info("Calculating in-dataset scores for %s", f)
            assert len(test_feature) == len(test_pred)
            for b in range(0, len(test_feature), batch_size):
                in_series[b : b + batch_size, i] = (
                    probability_weight_scores(
                        score_fn(test_feature[b : b + batch_size].to(DEVICE)),
                        pred=test_pred[b : b + batch_size],
                        probs=test_probs[b : b + batch_size].to(DEVICE),
                    )
                    .reshape(-1)
                    .cpu()
                )

            for out_dataset_name in args.out_dataset_names:
                logger.info("Calculating score for %s", out_dataset_name)
                config.update(out_dataset=out_dataset_name.upper())
                out_feature = torch.Tensor(res.np_load_out_features(config, f, args.reduction_op))
                logger.info("Out feature shape: %s", out_feature.shape)
                out_logits = torch.Tensor(res.np_load_out_logits(config))
                out_pred = torch.argmax(out_logits, dim=1)
                out_probs = torch.softmax(out_logits, dim=1)

                out_series[out_dataset_name] = out_series[out_dataset_name][: len(out_logits)]
                assert len(out_feature) == len(out_pred)
                for b in range(0, len(out_feature), batch_size):
                    out_series[out_dataset_name][b : b + batch_size, i] = (
                        probability_weight_scores(
                            score_fn(out_feature[b : b + batch_size].to(DEVICE)),
                            pred=out_pred[b : b + batch_size],
                            probs=out_probs[b : b + batch_size].to(DEVICE),
                        )
                        .reshape(-1)
                        .cpu()
                    )

                in_s = in_series[:, i].reshape(-1, 1).numpy()
                out_s = out_series[out_dataset_name][:, i].reshape(-1, 1).numpy()
                # layer performance calculation
                results = m.compute_detection_metrics(in_s, out_s)

                logger.info(
                    "Layer: %s, TNR: %s, AUROC: %s",
                    f,
                    round(results["tnr_at_0.95_tpr"], 3),
                    round(results["auroc"], 3),
                )

                save_obj = {
                    "model": config.nn_name,
                    "out_dataset": out_dataset_name,
                    "feature": f,
                    "reduction_op": args.reduction_op,
                    "score": config.score_method,
                    "auroc": results["auroc"],
                    "tnr_at_0.95_tpr": results["tnr_at_0.95_tpr"],
                }
                filename = os.path.join("results", "single_layer.csv")
                res.append_results_to_file(save_obj, filename=filename)

    # save train_series and in_series
    logger.info("Train series: %s, %s", train_series.shape, train_series.mean(0))
    logger.info("In series: %s, %s", in_series.shape, in_series.mean(0))

    train_series = train_series.numpy()
    res.np_save_train_timeseries(config, train_series, args.reduction_op)

    in_series = in_series.numpy()
    res.np_save_test_timeseries(config, in_series, args.reduction_op)

    for out_dataset_name in out_series.keys():
        config.update(out_dataset=out_dataset_name.upper())
        logger.info(
            "Out series %s: %s, %s",
            out_dataset_name,
            out_series[out_dataset_name].shape,
            out_series[out_dataset_name].mean(0),
        )
        out_series[out_dataset_name] = out_series[out_dataset_name].numpy()
        res.np_save_out_timeseries(config, out_series[out_dataset_name], args.reduction_op)


if __name__ == "__main__":
    args = Arguments(parser.parse_args())
    logging.basicConfig(
        format="--> %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    print(args)
    main(args)
