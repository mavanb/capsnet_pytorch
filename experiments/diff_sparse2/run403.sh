#!/usr/bin/env bash

# running on node 403

cd ../..

sparse_list=(
    none
    nodes_topk_0.3-0.3
    nodes_sample_0.3-0.3
    nodes_random_0.3-0.3
    edges_topk_0.3-0.3
#    edges_sample_0.3-0.3
#    edges_random_0.3-0.3
#    nodes_topk_0.2-0.2+edges_random_0.2-0.2
#    nodes_random_0.2-0.2+edges_random_0.2-0.2
#    nodes_topk_0.3-0.3+edges_random_0.3-0.3
#    nodes_sample_0.3-0.3+edges_sample_0.3-0.3
#    edges_random_0.0-0.3
    nodes_topk_0.0-0.3
#    edges_random_0.0-0.6
    nodes_topk_0.0-0.6
)


for seed in 1 2
do
    for dataset in cifar10 fashionmnist mnist
    do
        for sparse in "${sparse_list[@]}"
        do
            python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}: --general_conf experiments/diff_sparse/gen.conf --capsule_conf experiments/diff_sparse/caps.conf --seed $seed
        done
        python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_loss_01 --general_conf experiments/diff_sparse/gen.conf --capsule_conf experiments/diff_sparse/caps.conf --beta 0.1 --use_entropy True --seed $seed
        python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_loss_001 --general_conf experiments/diff_sparse/gen.conf --capsule_conf experiments/diff_sparse/caps.conf --beta 0.01 --use_entropy True --seed $seed
    done
done
