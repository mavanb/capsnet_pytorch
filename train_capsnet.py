from __future__ import print_function

import torch
from torchvision import transforms

from configurations.general_confs import get_conf, capsule_arguments
from data.data_loader import get_dataset
from ignite_features.trainer import CapsuleTrainer
from nets import BasicCapsNet
from loss import CapsuleLoss
from utils import get_logger


def main():
    # get arguments specific for capsule network
    custom_args = capsule_arguments("capsnet")

    # get general config
    conf, parser = get_conf(custom_args)

    # get logger and log config
    log = get_logger(__name__)
    log.info(parser.format_values())

    # get data set
    transform = transforms.ToTensor()
    data_train, data_test, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)

    model = BasicCapsNet(in_channels=data_shape[0], routing_iters=conf.routing_iters, in_height=data_shape[1],
                         in_width=data_shape[2], stdev_W=conf.stdev_W, bias_routing=conf.bias_routing,
                         sparse_threshold=conf.sparse_threshold, sparsify=conf.sparsify, sparse_topk=conf.sparse_topk,
                         arch=conf.architecture, recon=conf.use_recon)

    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, conf.beta, num_classes=label_shape,
                               include_recon=conf.use_recon, include_entropy=conf.use_entropy,
                               caps_sizes=model.caps_sizes)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    trainer = CapsuleTrainer(model, capsule_loss, optimizer, data_train, data_test, conf)

    trainer.run()


if __name__ == "__main__":
    main()
