#!/usr/bin/env bash

#

cd ../..

seed_arg=$1

sparse_list=(
    nodes_topk_0.3-0.3
    nodes_topk_0.0-0.6
    edges_random_0.3-0.3
    edges_topk_0.3-0.3
)

for seed_extra in 1 2
do
    seed=${seed_extra}${seed_arg}
    for sparse in "${sparse_list[@]}"
    do
        for dataset in cifar10 fashionmnist mnist
        do
            python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}:${seed}: --general_conf experiments/diff_sparse_new/gen.conf --capsule_conf experiments/diff_sparse_new/caps.conf --seed $seed
        done
        python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:loss_001:${seed}: --general_conf experiments/diff_sparse_new/gen.conf --capsule_conf experiments/diff_sparse_new/caps.conf --beta 0.01 --use_entropy True --seed $seed
    done
done



