#!/usr/bin/env bash

# This experiment check the effect of the number of routing iterations for various models.
# Model name should be formatted as:
#   $dataset:sparse_method:routing_iters:seed:double/single:
# Argument: node number. Is used as the seed.

cd ../..

seed=402

dataset=mnist
for routing_iters in 1 2 3
do
    python train_capsnet.py --dataset $dataset --routing_iters $routing_iters  --sparse none --model_name ${dataset}_:none:${routing_iters}:${seed}:single: --general_conf experiments/effect_routing_rerun/gen.conf --capsule_conf experiments/effect_routing_rerun/caps.conf --seed $seed
done