#!/bin/bash


## Test nodes topk for different sparsity degrees

# currently running on node 403
# use same config files as test_ent (Entropy testing)

python train_capsnet.py --model_name cifar_node_0.0\;0.3 --sparse_topk 0.0\;0.3 --dataset cifar10 --sparsify nodes_topk --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf
python train_capsnet.py --model_name cifar_node_0.3\;0.3 --sparse_topk 0.3\;0.3 --dataset cifar10 --sparsify nodes_topk --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf
python train_capsnet.py --model_name cifar_node_0.0\;0.6 --sparse_topk 0.0\;0.6 --dataset cifar10 --sparsify nodes_topk --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf
python train_capsnet.py --model_name cifar_node_0.3\;0.6 --sparse_topk 0.3\;0.6 --dataset cifar10 --sparsify nodes_topk --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf
python train_capsnet.py --model_name cifar_node_0.0\;0.9 --sparse_topk 0.0\;0.9 --dataset cifar10 --sparsify nodes_topk --general_conf configurations/gen_ent_test.conf --capsule_conf configurations/caps_ent_test.conf