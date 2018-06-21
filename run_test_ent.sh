#!/bin/bash


## Entropy testing

# currently running on node 432
# run file to do some test on the entropy
# to make sure that the same config is used, we use custom config files.

python train_capsnet.py --dataset fashionmnist --sparsify nodes_topk --model_name fashion_nodes_topk  --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf
python train_capsnet.py --dataset cifar10 --sparsify edges_random --model_name cifar_edges_random  --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf

python train_capsnet.py --dataset mnist --sparsify edges_topk --model_name cifar_edges_topk  --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf
python train_capsnet.py --dataset mnist --sparsify nodes_topk --model_name cifar_nodes_topk  --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf
python train_capsnet.py --dataset mnist --sparsify None --model_name cifar_None  --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf
python train_capsnet.py --dataset mnist --sparsify edges_random --model_name cifar_edges_random  --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf


