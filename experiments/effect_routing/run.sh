#!/usr/bin/env bash

# Argument: node number. Is used as the seed.

cd ../..

seed=$1

for dataset in mnist #fashionmnist # cifar10 shvn smallnorb
do
    for routing_iters in 1 2 3
    do
#        python train_capsnet.py --dataset $dataset --routing_iters $routing_iters  --sparse nodes_topk_0.0-0.6 --model_name ${dataset}_:nodes_topk_0.0-0.6:${routing_iters}:${seed}:double --general_conf experiments/effect_routing/gen.conf --capsule_conf experiments/effect_routing/caps.conf --seed $seed --architecture 32\,8\;10\,16\;10\,16
#        python train_capsnet.py --dataset $dataset --routing_iters $routing_iters  --sparse nodes_topk_0.0-0.6 --model_name ${dataset}_:nodes_topk_0.0-0.6:${routing_iters}:${seed} --general_conf experiments/effect_routing/gen.conf --capsule_conf experiments/effect_routing/caps.conf --seed $seed
        python train_capsnet.py --dataset $dataset --routing_iters $routing_iters  --sparse none --model_name ${dataset}_:none:${routing_iters}:${seed}:double --general_conf experiments/effect_routing/gen.conf --capsule_conf experiments/effect_routing/caps.conf --seed $seed --architecture 32\,8\;10\,16\;10\,16
    done
done

