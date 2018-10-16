"""Train basic capsule network

This module is used to train the basic capsule network. See the readme for a full list of all arguments.

Example:
        $ python train_capsnet.py

"""

from __future__ import print_function

import torch
from torchvision import transforms

from configurations.conf import get_conf, capsule_arguments
from data.data_loader import get_dataset
from ignite_features.trainer import CapsuleTrainer
from nets import BasicCapsNet
from loss import CapsuleLoss
from utils import get_logger
import numpy as np


def main():

    # get arguments specific for capsule network
    custom_args = capsule_arguments("capsnet")

    # get general config
    conf, parser = get_conf(custom_args)

    # get logger and log config
    log = get_logger(__name__)
    log.info(parser.format_values())

    # seed must be set before any stochastic operation in torch or numpy
    if conf.seed:
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

    # get data set
    transform = transforms.ToTensor()
    data_train, data_test, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)

    assert conf.architecture.final.caps == label_shape, "Number of final capsule should match the number of labels."

    # init basic capsnet
    model = BasicCapsNet(in_channels=data_shape[0], routing_iters=conf.routing_iters, in_height=data_shape[1],
                         in_width=data_shape[2], stdev_W=conf.stdev_W, bias_routing=conf.bias_routing,
                         arch=conf.architecture, recon=conf.use_recon, sparse=conf.sparse,
                         compute_activation=conf.compute_activation)

    # init capsule loss
    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, conf.beta, include_recon=conf.use_recon,
                               include_entropy=conf.use_entropy, caps_sizes=model.caps_sizes)

    # init adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    # init Trainer that supports the ignite training processs
    trainer = CapsuleTrainer(model, capsule_loss, optimizer, data_train, data_test, conf)

    # start trainer
    trainer.run()


if __name__ == "__main__":
    main()
