#!/bin/bash
# conda activate conda

python smoothed_infonce.py --n_epoch 6000 --alpha 0.001 --d 10

python smoothed_infonce.py --n_epoch 6000 --alpha 0.01 --d 10

python smoothed_infonce.py --n_epoch 6000 --alpha 0.1 --d 10