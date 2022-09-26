#!/bin/bash
source .env
python3 save_functional_dataset.py --detector projection \
    -outs TEXTURES MOS_PLACES MOS_SUN MOS_INATURALIST \
    --batch-size 50000 --model densenet121_ilsvrc2012

python3 save_functional_dataset.py --detector projection \
    -outs TEXTURES MOS_PLACES MOS_SUN MOS_INATURALIST \
    --batch-size 50000 --model bitsr101_ilsvrc2012

python3 save_functional_dataset.py --detector projection \
    -outs TEXTURES MOS_PLACES MOS_SUN MOS_INATURALIST \
    --batch-size 50000 --model vit16b_ilsvrc2012