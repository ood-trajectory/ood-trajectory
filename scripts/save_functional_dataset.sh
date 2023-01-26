#!/bin/bash
source .env
detectors=(
    "PROJECTION"\
)
models=(
    "DENSENET121"\
    "BITSR101"\
    "VIT16B"\
    "RESNET50"\
)
reduction_ops=(
    "adaptive-max-pool2d"\
)
for reduction in ${reduction_ops[*]}
do
    for model in ${models[*]}
    do
        for detector in ${detectors[*]}
        do
            python3 save_functional_dataset.py --detector $detector\
                -outs TEXTURES MOS_PLACES MOS_SUN MOS_INATURALIST \
                --batch-size 5000 --model ${model}_ilsvrc2012 --reduction_op $reduction
        done
    done
done
