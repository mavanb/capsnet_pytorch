from __future__ import print_function

import torch
from torchvision import transforms

from configurations.general_confs import parse_bool, get_conf
from data.data_loader import get_dataset
from ignite_features.trainer import CapsuleTrainer
from loss import CapsuleLoss
from nets import BasicCapsNet
from utils import get_logger


def custom_args(parser):
    parser.add('--basic_capsnet_config', is_config_file=True, default="configurations/basic_capsnet.conf",
               help='configurations file path')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model.')
    parser.add_argument('--alpha', type=float, required=True, help="Alpha of CapsuleLoss")
    parser.add_argument('--m_plus', type=float, required=True, help="m_plus of margin loss")
    parser.add_argument('--m_min', type=float, required=True, help="m_min of margin loss")
    parser.add_argument('--prim_caps', type=int, required=True, help="Number of primary capsules")
    parser.add_argument('--routing_iters', type=int, required=True,
                        help="Number of iterations in the routing algo.")
    parser.add_argument('--dataset', type=str, required=True, help="Either mnist or cifar10")
    parser.add_argument('--squash_dim', type=int, required=True, help="")
    parser.add_argument('--softmax_dim', type=int, required=True, help="")
    parser.add_argument('--stdev_W', type=float, required=True, help="stddev of W of capsule layer")
    parser.add_argument('--bias_routing', type=parse_bool, required=True, help="whether to use bias in routing")
    parser.add_argument('--excessive_testing', type=parse_bool, required=True,
                        help="Do excessive tests on test set")
    parser.add_argument('--sparse_threshold', type=float, required=True, help="Threshold of routing to sparsify.")
    parser.add_argument('--sparsify', type=str, required=True, help="The method used to sparsify the parse tree.")
    parser.add_argument('--sparse_topk', type=str, required=True, help="Percentage of non top k elements to exclude.")
    return parser


def main():
    conf, parser = get_conf(custom_args)
    log = get_logger(__name__)
    log.info(parser.format_values())
    transform = transforms.ToTensor()
    dataset, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)

    model = BasicCapsNet(in_channels=data_shape[0], digit_caps=label_shape, vec_len_prim=8, vec_len_digit=16,
                         routing_iters=conf.routing_iters, prim_caps=conf.prim_caps, in_height=data_shape[1],
                         in_width=data_shape[2], softmax_dim=conf.softmax_dim, squash_dim=conf.squash_dim,
                         stdev_W=conf.stdev_W, bias_routing=conf.bias_routing, sparse_threshold=conf.sparse_threshold,
                         sparsify=conf.sparsify, sparse_topk=conf.sparse_topk)

    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, num_classes=label_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = CapsuleTrainer(model, capsule_loss, optimizer, dataset, conf)

    trainer.run()


if __name__ == "__main__":
    main()
