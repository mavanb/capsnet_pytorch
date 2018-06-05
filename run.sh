#!/bin/bash

python caps_experiment.py --sparsify edges_topk --dataset mnist --model_name mnist_edges_topk
python caps_experiment.py --sparsify edges_topk --dataset fashionmnist --model_name fashionmnist_edges_topk
python caps_experiment.py --sparsify edges_topk --dataset cifar10 --model_name cifar10_edges_topk

python caps_experiment.py --sparsify edges_random --dataset mnist --model_name mnist_edges_random
python caps_experiment.py --sparsify edges_random --dataset fashionmnist --model_name fashionmnist_edges_random
python caps_experiment.py --sparsify edges_random --dataset cifar10 --model_name cifar10_edges_random

python caps_experiment.py --sparsify None --dataset mnist --model_name mnist_None
python caps_experiment.py --sparsify None --dataset fashionmnist --model_name fashionmnist_None
python caps_experiment.py --sparsify None --dataset cifar10 --model_name cifar10_None

python caps_experiment.py --sparsify nodes_topk --dataset mnist --model_name mnist_nodes_topk
python caps_experiment.py --sparsify nodes_topk --dataset fashionmnist --model_name fashionmnist_nodes_topk
python caps_experiment.py --sparsify nodes_topk --dataset cifar10 --model_name cifar10_nodes_topk

