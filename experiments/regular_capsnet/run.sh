#!/usr/bin/env bash

#

cd ../..

seed_arg=402

for seed_extra in 1 2 3
do
    seed=${seed_extra}${seed_arg}
    for dataset in cifar10 fashionmnist mnist
    do
        python train_capsnet.py --dataset $dataset --model_name ${dataset}_:none:${seed}: --general_conf experiments/regular_capsnet/gen.conf --capsule_conf experiments/regular_capsnet/caps.conf --seed $seed
    done
done



