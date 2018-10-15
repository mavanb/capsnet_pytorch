#!/usr/bin/env bash

#

cd ../..

seed_arg=407

sparse=none
dataset=cifar10

for seed_extra in 1 2 3
do
    seed=${seed_extra}${seed_arg}
    python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}:${seed}: --general_conf experiments/diff_sparse_rerun/gen.conf --capsule_conf experiments/diff_sparse_rerun/caps.conf --seed $seed
done



