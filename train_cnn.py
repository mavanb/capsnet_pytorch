from __future__ import print_function

import torch
from torch.nn.modules.loss import NLLLoss
from torchvision import transforms

from configurations.conf import get_conf
from data.data_loader import get_dataset
from ignite_features.trainer import CNNTrainer
from nets import BaselineCNN
from utils import get_logger


def custom_args(parser):
    parser.add('--baseline_cnn_config', is_config_file=True, default="configurations/baseline_cnn.conf",
          help='configurations file path')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model.')
    parser.add_argument('--dataset', type=str, required=True, help="Either mnist or cifar10")
    parser.add_argument('--stdev_W', type=float, required=True, help="stddev of W of capsule layer")
    return parser


def main():

    conf, parser = get_conf(custom_args)
    log = get_logger(__name__)
    log.info(parser.format_values())

    transform = transforms.ToTensor()
    dataset, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)

    model = BaselineCNN(classes=10, in_channels=data_shape[0], in_height=data_shape[1], in_width=data_shape[2])

    nlll = NLLLoss()

    optimizer = torch.optim.Adam(model.parameters())

    trainer = CNNTrainer(model, nlll, optimizer, dataset, conf)

    trainer.run()


if __name__ == "__main__":
    main()


