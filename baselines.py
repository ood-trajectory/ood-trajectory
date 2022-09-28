import os
import sys

import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm

import data
import eval.metrics as m
import eval.results as res
import nn as models

torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def iterate_data_odin(data_loader, model, epsilon, temper):
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    confs = []
    for b, (x, y) in enumerate(tqdm(data_loader, "Odin")):
        x = Variable(x.to(DEVICE), requires_grad=True)
        outputs = model(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper
        if epsilon > 0:
            labels = Variable(torch.LongTensor(maxIndexTemp).to(DEVICE))
            loss = criterion(outputs, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(x.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            # Adding small perturbations to images
            x = torch.add(x.data, -epsilon, gradient)
            outputs = model(Variable(x))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs / temper

        outputs = outputs.data.cpu()
        outputs = outputs.numpy()
        outputs = outputs - np.max(outputs, axis=1, keepdims=True)
        outputs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)

        confs.extend(np.max(outputs, axis=1))

    return np.array(confs)


def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(tqdm(data_loader, "Energy")):
        with torch.no_grad():
            x = x.to(DEVICE)
            # compute output, measure accuracy and record loss.
            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            conf = temper * torch.logsumexp(outputs / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_gradnorm(
    data_loader, model, temperature, num_classes, model_type="bit"
):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(DEVICE)
    for b, (x, y) in enumerate(tqdm(data_loader, "GradNorm")):
        inputs = Variable(x.to(DEVICE), requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        targets = torch.ones((inputs.shape[0], num_classes)).to(DEVICE)
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()
        if model_type == "bit":
            layer_grad = model.head.conv.weight.grad.data
        elif model_type == "dn":
            layer_grad = model.classifier.weight.grad.data
        else:
            layer_grad = model.heads.head.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)


def run_eval(model, in_loader, out_loader):
    # switch to evaluate mode
    model.eval()

    if args.detector == "ODIN":
        in_scores = iterate_data_odin(in_loader, model, args.eps, args.temperature)
        out_scores = iterate_data_odin(out_loader, model, args.eps, args.temperature)
    elif args.detector == "ENERGY":
        in_scores = iterate_data_energy(in_loader, model, args.temperature)
        out_scores = iterate_data_energy(out_loader, model, args.temperature)
    elif args.detector == "GRADNORM":
        if "bit" in args.model.lower():
            model_type = "bit"
        elif "densenet" in args.model.lower():
            model_type = "dn"
        else:
            model_type = "vit"

        in_scores = iterate_data_gradnorm(
            in_loader, model, args.temperature, num_classes, model_type=model_type
        )
        out_scores = iterate_data_gradnorm(
            out_loader, model, args.temperature, num_classes, model_type=model_type
        )

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    results_path = os.path.join("results", "results_baselines.csv")
    results1 = m.compute_detection_metrics(in_examples, out_examples)
    config.update(**results1)
    save_obj = {
        "nn_name": config.nn_name,
        "out_dataset": args.dataset,
        "score": args.detector,
        "temperature": args.temperature,
        "eps": args.eps,
        "fpr_at_0.95_tpr": results1["fpr_at_0.95_tpr"],
        "tnr_at_0.95_tpr": results1["tnr_at_0.95_tpr"],
        "auroc": results1["auroc"],
    }
    res.append_results_to_file(save_obj, results_path)

    results2 = m.compute_detection_metrics(-in_examples, -out_examples)
    config.update(**results1)
    save_obj = {
        "nn_name": config.nn_name,
        "out_dataset": args.dataset,
        "score": args.detector,
        "temperature": args.temperature,
        "eps": args.eps,
        "fpr_at_0.95_tpr": results2["fpr_at_0.95_tpr"],
        "tnr_at_0.95_tpr": results2["tnr_at_0.95_tpr"],
        "auroc": results2["auroc"],
    }
    res.append_results_to_file(save_obj, results_path)
    print(config)


def main():
    if args.detector == "gradnorm":
        args.batch_size = 1

    transform_test = models.get_data_transform(args.model, train=False)
    in_dataset = data.get_dataset(
        in_dataset_name, transform=transform_test, train=False
    )
    in_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=args.batch_size, shuffle=False
    )
    out_dataset = data.get_dataset(args.dataset, transform=transform_test, train=False)
    out_loader = torch.utils.data.DataLoader(
        out_dataset, batch_size=args.batch_size, shuffle=False
    )
    print("Datasets loaded.")

    model = models.get_pre_trained_model_by_name(args.model, pre_trained=True)
    model = model.to(DEVICE)
    print("Model loaded.")
    run_eval(model, in_loader, out_loader)


if __name__ == "__main__":
    from argparser import Arguments, parser

    args = Arguments(parser.parse_args())
    in_dataset_name = models.get_in_dataset_name(args.model)
    if in_dataset_name.upper() == args.dataset.upper():
        print("In dataset is equal to Out dataset.")
        sys.exit(1)
    num_classes = models.get_num_classes(args.model)

    config = res.Config(
        nn_name=args.model,
        dataset_name=in_dataset_name,
        out_dataset=args.dataset,
        score_method=args.detector,
    )
    print(config)
    main()
