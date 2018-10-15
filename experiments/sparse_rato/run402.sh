#!/usr/bin/env bash


cd ../..

seed_arg=402
dataset=cifar10

sparse_list=(
    nodes_topk_0.0-0.5
)

for sparse in "${sparse_list[@]}"
do
    for seed_extra in 1 2 3
    do
            seed=${seed_extra}${seed_arg}
            python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}:${seed}: --general_conf experiments/sparse_rato/gen.conf --capsule_conf experiments/sparse_rato/caps.conf --seed $seed
    done
done