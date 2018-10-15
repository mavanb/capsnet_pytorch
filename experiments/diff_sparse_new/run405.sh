#!/usr/bin/env bash

cd ../..

seed_arg=405

## Load and continue mnist_:nodes_topk_0.3-0.3:1405

#python train_capsnet.py --dataset mnist --sparse nodes_topk_0.3-0.3 --model_name mnist_:nodes_topk_0.3-0.3:1405: --general_conf experiments/diff_sparse_new/gen.conf --capsule_conf experiments/diff_sparse_new/caps.conf --seed 1405 --load_model True --load_name mnist_:nodes_topk_0.3-0.3:1405:_best_sparse__61.pth


### Rerun the remaining MNIST. 2 required, thus on both 402 and 405

sparse_list=(
    edges_random_0.3-0.3
    edges_topk_0.3-0.3
    none
)

seed_extra=2
seed=${seed_extra}${seed_arg}

# rerun all sparse mnist except entropy
for sparse in "${sparse_list[@]}"
do
    for dataset in mnist
    do
        python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}:${seed}: --general_conf experiments/diff_sparse_new/gen.conf --capsule_conf experiments/diff_sparse_new/caps.conf --seed $seed
    done
done

## Cifar10 and fashion, All that required 2 more are runned on 402 and 405 (here)
# none only runned here, as 1 required

sparse_list=(
    nodes_topk_0.0-0.6
    edges_random_0.3-0.3
    edges_topk_0.3-0.3
    none
)

seed=${seed_extra}${seed_arg}

python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}:${seed}: --general_conf experiments/diff_sparse_new/gen.conf --capsule_conf experiments/diff_sparse_new/caps.conf --seed $seed

# rerun all sparse for all datasets
for sparse in "${sparse_list[@]}"
do
    for dataset in cifar10 fashionmnist
    do
        python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:${sparse}:${seed}: --general_conf experiments/diff_sparse_new/gen.conf --capsule_conf experiments/diff_sparse_new/caps.conf --seed $seed
    done
done

### rerun entropy for fashion and cifar. 3 required, so 2 on 402, 1 on 405
for dataset in cifar10 fashionmnist
do
    python train_capsnet.py --dataset $dataset --sparse $sparse --model_name ${dataset}_:loss_001:${seed}: --general_conf experiments/diff_sparse_new/gen.conf --capsule_conf experiments/diff_sparse_new/caps.conf --beta 0.01 --use_entropy True --seed $seed
done


