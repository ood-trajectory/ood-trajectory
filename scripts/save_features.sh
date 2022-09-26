#!/bin/bash
source .env
in_dataset_name="ILSVRC2012"
models=(
    "DENSENET121"\
    "BITSR101"\
    "VIT16B"\
)
out_dataset_names=(
    "TEXTURES"\
    "MOS_INATURALIST"\
    "MOS_SUN"\
    "MOS_PLACES"\
)
for model in ${models[*]}
do
    model_name=${model}_${in_dataset_name}
    python3 save_features.py \
        --model $model_name  \
        --dataset $in_dataset_name

    python3 save_features.py \
        --model $model_name  \
        --dataset $in_dataset_name --train
    
    for out_dataset_name in ${out_dataset_names[*]}
    do
        python3 save_features.py \
            --model $model_name \
            --dataset $out_dataset_name
    done
done
