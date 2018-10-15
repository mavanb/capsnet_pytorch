#!/usr/bin/env bash

cd ../..

seed_arg=401

for seed_extra in 1 2 3
do
    seed=${seed_extra}${seed_arg}
    python train_capsnet.py --model_name mnist_:old_loss:${seed}: --general_conf experiments/special_test/gen.conf --capsule_conf experiments/special_test/caps.conf --seed $seed --beta 0.1
done


