#!/usr/bin/env bash

# This experiment check the effect of the number of routing iterations for various models.
# Model name should be formatted as:
#   $dataset:sparse_method:routing_iters:seed:double/single:
# Argument: node number. Is used as the seed.

cd ../..

seed=$1

for dataset in mnist
do
    for routing_iters in 1 2 3
    do
        python train_capsnet.py --dataset $dataset --routing_iters $routing_iters  --sparse nodes_topk_0.3-0.3 --model_name ${dataset}_:nodes_topk_0.3-0.3:${routing_iters}:${seed}:double: --general_conf experiments/effect_routing_new/gen.conf --capsule_conf experiments/effect_routing_new/caps.conf --seed $seed --architecture 32\,8\;10\,16\;10\,16
    done
done

