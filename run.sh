#!/bin/bash

# hidden capsules test
python caps_experiment.py --sparsify edges_topk --hidden_capsules 200 --dataset cifar10 --model_name hidden_200_topk
python caps_experiment.py --sparsify None --hidden_capsules 200 --dataset cifar10 --model_name hidden_200_None

python caps_experiment.py --sparsify None --hidden_capsules 100 --dataset cifar10 --model_name hidden_100_None
python caps_experiment.py --sparsify edges_topk --hidden_capsules 100 --dataset cifar10 --model_name hidden_100_topk

python caps_experiment.py --sparsify None --hidden_capsules 50 --dataset cifar10 --model_name hidden_50_None
python caps_experiment.py --sparsify edges_topk --hidden_capsules 50 --dataset cifar10 --model_name hidden_50_topk

python caps_experiment.py --sparsify edges_topk --hidden_capsules 10 --dataset cifar10 --model_name hidden_10_topk
python caps_experiment.py --sparsify None --hidden_capsules 10 --dataset cifar10 --model_name hidden_10_None



