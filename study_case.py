import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import torch

import data as data_utils
import eval.metrics as m
import eval.results as res

plt.style.use(["science", "light", "no-latex"])
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def energy_score(logits, temperature=1.0):
    return temperature * torch.log(torch.sum(torch.exp(logits / temperature), dim=1))


def main_textures():
    model_name = "DENSENET121_ILSVRC2012"
    in_dataset_name = "ILSVRC2012"
    dataset_name = "TEXTURES"
    reduction_op = "adaptive-max-pool2d"

    config = res.Config(
        nn_name=model_name,
        dataset_name=in_dataset_name,
        out_dataset=dataset_name,
        score_method="PROJECTION",
        aggregation="innerproduct",
    )

    TENSORS_DIR = os.environ.get("TENSORS_DIR", "./tensors")

    out_targets = np.load(
        os.path.join(TENSORS_DIR, model_name, dataset_name, "targets.npy")
    )
    out_logits = np.load(
        os.path.join(TENSORS_DIR, model_name, dataset_name, "logits.npy")
    )
    in_targets = np.load(
        os.path.join(TENSORS_DIR, model_name, in_dataset_name, "targets.npy")
    )
    in_logits = np.load(
        os.path.join(TENSORS_DIR, model_name, in_dataset_name, "logits.npy")
    )

    temperature = 1
    in_energy = energy_score(
        torch.from_numpy(in_logits), temperature=temperature
    ).numpy()
    out_energy = energy_score(
        torch.from_numpy(out_logits), temperature=temperature
    ).numpy()
    results_energy = m.compute_detection_metrics(in_energy, out_energy)

    in_traj = res.np_load_test_timeseries(config, reduction_op)
    out_traj = res.np_load_out_timeseries(config, reduction_op)
    in_scores = res.np_load_in_scores(config, reduction_op)
    out_scores = res.np_load_out_scores(config, reduction_op)
    results = m.compute_detection_metrics(in_scores, out_scores)

    print("Energy score")
    print(json.dumps(results_energy, indent=2))
    print("Ours")
    print(json.dumps(results, indent=2))

    energy_thr = 10.881546020507812
    ours_thr = 0.8704515099525452

    # confusion matrix
    energy = np.concatenate([in_energy, out_energy])
    energy_labels = np.concatenate([np.zeros(len(in_energy)), np.ones(len(out_energy))])
    energy_pred = energy <= energy_thr
    print("Energy confusion matrix")
    print(skm.confusion_matrix(energy_labels, energy_pred))
    tn, fp, fn, tp = skm.confusion_matrix(energy_labels, energy_pred).ravel()
    print(tn, fp, fn, tp)

    ours = np.concatenate([in_scores, out_scores])
    ours_labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    ours_pred = ours <= ours_thr
    print("Ours confusion matrix")
    print(skm.confusion_matrix(ours_labels, ours_pred))
    tn, fp, fn, tp = skm.confusion_matrix(ours_labels, ours_pred).ravel()
    print(tn, fp, fn, tp)

    # finter false negatives
    print("Energy false negatives")
    energy_filt = energy_pred[len(in_energy) :] == 0
    energy_fns = out_targets[energy_filt]
    energy_fns = energy_fns.astype(np.int64)

    print("Ours false negatives")
    ours_filt = ours_pred[len(in_scores) :] == 0
    ours_fns = out_targets[ours_filt]
    ours_fns = ours_fns.astype(np.int64)

    # class frequency
    energy_cf = {i: np.bincount(energy_fns)[i] for i in range(47)}
    ours_cf = {i: np.bincount(ours_fns)[i] for i in range(47)}

    ## sort
    energy_cf = {
        k: v
        for k, v in sorted(energy_cf.items(), key=lambda item: item[1], reverse=True)
    }
    ours_cf = {
        k: v for k, v in sorted(ours_cf.items(), key=lambda item: item[1], reverse=True)
    }
    print("Energy class frequency")
    print(energy_cf)
    print("Ours class frequency")
    print(ours_cf)

    # 120 images per class in Textures
    # honeycombed: textures and honeycomb imagenet (599)
    HC = 18  # from 108 mistakes to 20

    # stripes: textures and zebra imagenet (340)
    ST = 39  # from 94 mistakes to 16

    # cobwebbed: textures and spider web imagenet (815)
    CW = 6  # from 101 to 52 mistakes
    # image examples
    dataset = data_utils.get_dataset(dataset_name)

    # get image by class
    cl = [HC, ST, CW]
    idxs = {HC: [], ST: [], CW: []}
    for cl_ref in cl:
        counter = 0
        for i, (image, label) in enumerate(dataset):
            if label == cl_ref:
                idxs[cl_ref].append(i)
                im = image

                # plot image
                # plt.imshow(im)
                os.makedirs("images/textures", exist_ok=True)
                im.save(f"images/textures/{cl_ref}_{counter}.png")
                counter += 1
                if counter > 4:
                    break
    print(idxs)

    dataset = data_utils.get_dataset(in_dataset_name)
    cl = [599, 340, 815, 292, 290]
    idxs_in = {599: [], 340: [], 292: [], 290: [], 815: []}
    for cl_ref in cl:
        counter = 0
        for i, (image, label) in enumerate(dataset):
            if label == cl_ref:
                idxs_in[cl_ref].append(i)
                im = image

                # plot image
                # plt.imshow(im)
                os.makedirs("images/imagenet", exist_ok=True)

                im.save(f"images/imagenet/{cl_ref}_{counter}.png")
                counter += 1
                if counter > 2:
                    break

    # approx. equal
    max_scale = np.max(in_traj, axis=0)
    in_traj = in_traj / max_scale

    trajectories = defaultdict(list)
    traj_scores = defaultdict(list)
    for cl_ref in [HC, ST, CW]:
        for i in idxs[cl_ref]:
            trajectories[cl_ref].append(out_traj[i] / max_scale)
            traj_scores[cl_ref].append(out_scores[i])

    print("Scores out", traj_scores)

    trajectories_in = defaultdict(list)
    traj_scores_in = defaultdict(list)
    for cl_ref in [599, 340, 815, 292, 290]:
        for i in idxs_in[cl_ref]:
            trajectories_in[cl_ref].append(in_traj[i] / max_scale)
            traj_scores_in[cl_ref].append(in_scores[i])

    print("Scores in", traj_scores_in)

    average_traj = np.mean(in_traj, axis=0)
    traj_ood_honeycomb = out_traj[idxs[HC][0]] / max_scale
    traj_ood_stripes = out_traj[idxs[ST][1]] / max_scale
    traj_ood_cobwebbed = out_traj[idxs[CW][2]] / max_scale

    # plot
    plt.figure(figsize=(4, 3))
    plt.plot(average_traj, label="Reference trajectory", color="b", marker="+")
    plt.plot(traj_ood_honeycomb, label="Honeycomb (0.61)", color="green", marker="*")
    plt.plot(traj_ood_stripes, label="Stripes (0.83)", color="green", marker="o")
    plt.plot(traj_ood_cobwebbed, label="Cobwebbed (0.89)", color="red", marker="^")
    plt.legend(frameon=1)
    plt.xlabel("Feature")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("images/traj.pdf", dpi=1000, bbox_inches="tight")

    # innerproduct
    s1 = np.sum(average_traj * traj_ood_honeycomb) / np.sum(average_traj**2)
    s2 = np.sum(average_traj * traj_ood_stripes) / np.sum(average_traj**2)
    s3 = np.sum(average_traj * traj_ood_cobwebbed) / np.sum(average_traj**2)
    print(s1, s2, s3)


if __name__ == "__main__":
    main_textures()
