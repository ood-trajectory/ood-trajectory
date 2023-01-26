import logging
import os
import random
import sys

import numpy as np

import eval.metrics as m
import eval.results as res
from aggregations import aggs
from argparser import Arguments, parser

logger = logging.getLogger(__name__)


class MaxScaling:
    def __init__(self) -> None:
        self.max = None

    def fit(self, X: np.ndarray, *args, **kwargs):
        self.max = X.max(axis=0, keepdims=True)

    def transform(self, x: np.ndarray, *args, **kwargs):
        return x / (self.max)

    def fit_transform(self, X: np.ndarray, *args, **kwargs):
        self.fit(X, *args, **kwargs)
        return self.transform(X, *args, **kwargs)


def main(args: Arguments):
    random.seed(args.seed)
    np.random.seed(args.seed)

    in_dataset_name = args.model.split("_")[1]
    config = res.Config(
        nn_name=args.model,
        dataset_name=in_dataset_name,
        out_dataset=args.dataset,
        score_method=args.detector,
        aggregation=args.aggregation,
    )

    """load features"""
    train_logits = res.np_load_train_logits(config)

    train_pred = np.argmax(train_logits, axis=1)

    in_logits = res.np_load_test_logits(config)
    in_pred = np.argmax(in_logits, axis=1)

    train_time_series = res.np_load_train_timeseries(config, args.reduction_op)
    X_train = train_time_series
    logger.info("X_train.shape: %s", X_train.shape)

    test_time_series = res.np_load_test_timeseries(config, args.reduction_op)
    logger.info("test_time_series.shape: %s", test_time_series.shape)

    scaler = MaxScaling()
    X_train = scaler.fit_transform(X_train, train_pred)

    agg = aggs.get_aggregation(args.aggregation, sample_size=256, ntrees=10)
    agg.fit(X_train, pred=train_pred)

    aurocs = []
    tnrs = []
    for name in args.out_dataset_names:
        name = name.upper()
        config.update(out_dataset=name)
        out_logits = res.np_load_out_logits(config)

        out_pred = np.argmax(out_logits, axis=1)

        test_pred = np.concatenate([in_pred, out_pred])
        out_time_series = res.np_load_out_timeseries(config, args.reduction_op)

        print("---------------------------")
        print("Dataset: ", name)
        logger.info("train_time_series shape: %s", train_time_series.shape)
        logger.info("test_time_series shape: %s", test_time_series.shape)
        logger.info("out_time_series shape: %s", out_time_series.shape)

        max_auroc = 0
        for i in range(test_time_series.shape[1]):
            in_scores = test_time_series[:, i]
            out_scores = out_time_series[:, i]
            results1 = m.compute_detection_metrics(in_scores, out_scores)
            this_auroc = results1["auroc"]
            print(f"{i}: AUROC: {this_auroc:.4f}", end=" ")
            if this_auroc > max_auroc:
                max_auroc = this_auroc

        print()
        print(f"Best AUROC: {max_auroc:.4f}")

        X_test = np.concatenate([test_time_series, out_time_series], axis=0)
        y_test = np.concatenate(
            [np.ones(test_time_series.shape[0]), np.zeros(out_time_series.shape[0])],
            axis=0,
        )
        y_pred = np.concatenate([in_pred, out_pred])
        assert X_test.shape[0] == y_test.shape[0] == y_pred.shape[0]
        X_test = scaler.transform(X_test, y_pred)
        scores = agg.forward(X_test, pred=test_pred)

        assert len(scores) == len(X_test)

        in_scores = scores[y_test == 1]
        out_scores = scores[y_test == 0]
        results1 = m.compute_detection_metrics(in_scores, out_scores)
        print(f"TNR: {results1['tnr_at_0.95_tpr']:.4f}, AUROC: {results1['auroc']:.4f}")

        # save results
        save_obj = {
            "model": config.nn_name,
            "out_dataset": config.out_dataset,
            "reduction_op": args.reduction_op,
            "score": config.score_method,
            "agg": args.aggregation,
            "auroc": results1["auroc"],
            "tnr_at_0.95_tpr": results1["tnr_at_0.95_tpr"],
        }
        filename = os.path.join("results", "main_results.csv")
        res.append_results_to_file(save_obj, filename=filename)

        aurocs.append(results1["auroc"])
        tnrs.append(results1["tnr_at_0.95_tpr"])

        # save scores
        res.np_save_in_scores(config, in_scores, args.reduction_op)
        res.np_save_out_scores(config, out_scores, args.reduction_op)

    # save average results
    auroc = np.mean(aurocs)
    tnr = np.mean(tnrs)
    print(f"AVERAGE TNR: {tnr:.4f}, AUROC: {auroc:.4f}")
    save_obj = {
        "model": config.nn_name,
        "out_dataset": "AVERAGE",
        "reduction_op": args.reduction_op,
        "score": config.score_method,
        "agg": args.aggregation,
        "auroc": auroc,
        "tnr_at_0.95_tpr": tnr,
    }
    filename = os.path.join("results", "main_results.csv")
    res.append_results_to_file(save_obj, filename=filename)


if __name__ == "__main__":
    args = Arguments(parser.parse_args())
    logging.basicConfig(
        format="--> %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main(args)
