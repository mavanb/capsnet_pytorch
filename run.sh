#!/bin/bash

# hidden capsules test
#python train_capsnet.py --dataset fashionmnist --sparsify edges_topk --model_name fashion_edges_topk
#python train_capsnet.py --dataset fashionmnist --sparsify nodes_topk --model_name fashion_nodes_topk
#python train_capsnet.py --dataset fashionmnist --sparsify None --model_name fashion_None
python train_capsnet.py --dataset fashionmnist --sparsify edges_random --model_name fashion_edges_random

#python train_capsnet.py --dataset cifar10 --sparsify edges_topk --model_name cifar_edges_topk
#python train_capsnet.py --dataset cifar10 --sparsify nodes_topk --model_name cifar_nodes_topk
#python train_capsnet.py --dataset cifar10 --sparsify None --model_name cifar_None
#python train_capsnet.py --dataset cifar10 --sparsify edges_random --model_name cifar_edges_random



