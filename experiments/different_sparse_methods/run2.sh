#!/usr/bin/env bash

# running on node 403

cd ../..

# loop over the sparsify methods
for seed in 1 2
do
    for dataset in cifar10
    do
       for sparse in edges_random edges_topk nodes_topk None
       do
            python train_capsnet.py --dataset $dataset --sparsify $sparse --model_name ${dataset}_${sparse} --general_conf experiments/different_sparse_methods/gen_ent_test.conf --capsule_conf experiments/different_sparse_methods/caps_ent_test.conf --seed $seed
       done
       python train_capsnet.py --dataset $dataset --sparsify None --model_name ${dataset}_loss_01 --general_conf experiments/different_sparse_methods/gen_ent_test.conf --capsule_conf experiments/different_sparse_methods/caps_ent_test.conf --beta 0.1 --use_entropy True --seed $seed
       python train_capsnet.py --dataset $dataset --sparsify None --model_name ${dataset}_loss_001 --general_conf experiments/different_sparse_methods/gen_ent_test.conf --capsule_conf experiments/different_sparse_methods/caps_ent_test.conf --beta 0.01 --use_entropy True --seed $seed
    done
done

