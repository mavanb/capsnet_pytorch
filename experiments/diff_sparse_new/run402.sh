#!/usr/bin/env bash

cd ../..

seed_arg=402

#### Rerun the remaining MNIST. 2 required, thus on both 402 and 405
#
#sparse_list=(
#    edges_random_0.3-0.3
#    edges_topk_0.3-0.3
#    none
#)
#
#seed_extra=2
#seed=${seed_extra}${seed_arg}
#
## rerun all sparse mnist except entropy
#for sparse in "${sparse_list[@]}"
#do
#    for dataset in mnist
#    do
#        python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}:${seed}: --general_conf experiments/diff_sparse_new/gen.conf --capsule_conf experiments/diff_sparse_new/caps.conf --seed $seed
#    done
#done

## Cifar10 and fashion, All that required 2 more are runned on 402 (here) and 405

#sparse_list=(
#    nodes_topk_0.0-0.6
#    nodes_topk_0.3-0.3
#)
#

#
## rerun some for cifar and fashion
#for sparse in "${sparse_list[@]}"
#do
#    for dataset in mnist
#    do
#        python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}:${seed}: --general_conf experiments/diff_sparse_new/gen.conf --capsule_conf experiments/diff_sparse_new/caps.conf --seed $seed
#    done
#done

dataset=cifar10
sparse=nodes_random_0.3-0.3

for seed_extra in 1 2 3
do
    seed=${seed_extra}${seed_arg}
    python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}:${seed}: --general_conf experiments/diff_sparse_new/gen.conf --capsule_conf experiments/diff_sparse_new/caps.conf --seed $seed
done





