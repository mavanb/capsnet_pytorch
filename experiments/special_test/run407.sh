#!/usr/bin/env bash

cd ../..

seed_arg=407

for seed_extra in 1 2 3
do
    seed=${seed_extra}${seed_arg}
    python train_capsnet.py --model_name mnist_:new_loss:${seed}: --general_conf experiments/special_test/gen.conf --capsule_conf experiments/special_test/caps.conf --seed $seed --beta -1.0
done

