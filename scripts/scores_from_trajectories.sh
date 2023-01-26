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
    for agg in "innerproduct" "mahalanobis" "EUCLIDES" "ONE_CLASS_SVM" "IFOREST"
    do
        for model in ${models[*]}
        do
            for detector in ${detectors[*]}
            do
                python3 trajectory_score.py --detector $detector\
                    -outs TEXTURES MOS_PLACES MOS_SUN MOS_INATURALIST \
                    --model ${model}_ilsvrc2012 \
                    -agg $agg --reduction_op $reduction
            done
        done
    done
done
