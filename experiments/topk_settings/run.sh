#!/usr/bin/env bash

$root_folder && python train_capsnet.py --model_name cifar_random_0.4\;0.4 --sparse_topk 0.4\;0.4 --dataset cifar10 --sparsify edges_random --general_conf configurations/gen_ent_loss.conf --capsule_conf configurations/caps_ent_loss.conf
$root_folder && python train_capsnet.py --model_name cifar_random_0.3\;0.3 --sparse_topk 0.3\;0.3 --dataset cifar10 --sparsify edges_random --general_conf configurations/gen_ent_loss.conf --capsule_conf configurations/caps_ent_loss.conf
$root_folder && python train_capsnet.py --model_name cifar_random_0.0\;0.3 --sparse_topk 0.0\;0.3 --dataset cifar10 --sparsify edges_random --general_conf configurations/gen_ent_loss.conf --capsule_conf configurations/caps_ent_loss.conf
$root_folder && python train_capsnet.py --model_name cifar_random_0.0\;0.6 --sparse_topk 0.0\;0.6 --dataset cifar10 --sparsify edges_random --general_conf configurations/gen_ent_loss.conf --capsule_conf configurations/caps_ent_loss.conf
$root_folder && python train_capsnet.py --model_name cifar_nodes_0.4\;0.4_40epoch --sparse_topk 0.4\;0.4 --dataset cifar10 --sparsify nodes_topk --general_conf configurations/gen_ent_loss.conf --capsule_conf configurations/caps_ent_loss.conf --epochs 40



