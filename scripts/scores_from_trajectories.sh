#!/bin/bash
source .env
python3 trajectory_score.py --detector projection \
    -outs TEXTURES MOS_PLACES MOS_SUN MOS_INATURALIST \
    --model densenet121_ilsvrc2012 \
    -agg innerproduct

python3 trajectory_score.py --detector projection \
    -outs TEXTURES MOS_PLACES MOS_SUN MOS_INATURALIST \
    --model bitsr101_ilsvrc2012 \
    -agg innerproduct

python3 trajectory_score.py --detector projection \
    -outs TEXTURES MOS_PLACES MOS_SUN MOS_INATURALIST \
    --model vit16b_ilsvrc2012 \
    -agg innerproduct