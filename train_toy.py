from __future__ import print_function

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
import configargparse

from data.gaussian2d import Gaussian2D
from loss import CapsuleLoss
from nets import ToyCapsNet
from utils import get_device
from utils import get_logger
from configurations.conf import SparseMethods, parse_bool


def toy_args(parser):
    """

    Args:
        parser:

    Returns:

    """
    parser.add('--toy_capsnet_config', is_config_file=True, default="configurations/toy_capsnet.conf",
               help='configurations file path')
    parser.add_argument('--output_file', type=str, required=True, help='Name of output file (routing points pickle)')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder in experiments folder')
    parser.add_argument('--alpha', type=float, required=True, help="Alpha of CapsuleLoss")
    parser.add_argument('--data_dim', type=int, required=True, help="Dimensions of the input data.")
    parser.add_argument('--m_plus', type=float, required=True, help="m_plus of margin loss")
    parser.add_argument('--m_min', type=float, required=True, help="m_min of margin loss")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs.")
    parser.add_argument('--routing_iters', type=int, required=True, help="Number of iterations in the routing algo.")
    parser.add_argument('--bias_routing', type=parse_bool, required=True, help="whether to use bias in routing")
    parser.add_argument('--use_recon', type=parse_bool, required=True, help="Use reconstruction yes/no.")
    parser.add_argument('--sparse', type=SparseMethods, required=True, help="Sparsity procedure.")
    return parser


class RoutingPoint:
    def __init__(self, u_jin):
        self.u_jin = u_jin

        self.acc = None
        self.epoch = None
        self.iter = None
        self.label = None
        self.latent = None
        self.routing_iters = []


class RoutingIter:
    def __init__(self, routing_iter, c_ji, s_jn, v_jn):
        self.routing_iter = routing_iter
        self.c_ji = c_ji
        self.s_jn = s_jn
        self.v_jn = v_jn


def main():

    def log_function(index, u_hat, b_vec, c_vec, v_vec, s_vec, s_vec_bias):

        # if first routing iterations, append routing point to list
        if index == 0:
            routing_point = RoutingPoint(u_hat.data.cpu().numpy() if torch.cuda.is_available() else u_hat.data.numpy())
            routing_point_list.append(routing_point)

        if torch.cuda.is_available():
            routing_iter = RoutingIter(index, c_vec.data.cpu().numpy(), s_vec.data.cpu().numpy(),
                                       v_vec.data.cpu().numpy())
        else:
            routing_iter = RoutingIter(index, c_vec.data.numpy(), s_vec.data.numpy(),
                                       v_vec.data.numpy())

        # add routing point to last element of the list
        routing_point_list[-1].routing_iters.append(routing_iter)

    def train(batch, epoch, iter):

        model.rout_layer.log_function = log_function

        observed, label, latent = batch

        model.train()
        optimizer.zero_grad()

        data = batch[0].to(get_device())
        labels = batch[1].to(get_device())

        logits, recon, _ = model(data, labels)

        total_loss, margin_loss, recon_loss, _ = capsule_loss(data, labels, logits, recon)

        # current routing point is the last in list
        routing_point = routing_point_list[-1]

        total_loss.backward()
        optimizer.step()

        print(f"\rLoss: {total_loss.item():0.4f}, Margin loss: {margin_loss.item():0.4f}, Recon loss: {conf.alpha * recon_loss.item():0.4f}", end="")

        # evaluate at every training step
        acc = evaluate()

        # add all training step info to the routing point
        routing_point.acc = acc
        routing_point.iter = iter
        routing_point.epoch = epoch
        routing_point.latent = latent
        routing_point.label = label
        routing_point_list.append(routing_point)

        return acc

    def evaluate():

        model.eval()

        # do not log during eval
        model.rout_layer.log_function = None

        with torch.no_grad():

            acc_values = []
            for batch_idx, batch in enumerate(val_loader):

                data = batch[0].to(get_device())
                labels = batch[1].to(get_device())

                logits, recon, _ = model(data)
                loss, _, _, _ = capsule_loss(data, labels, logits, recon)

                acc = model.compute_acc(logits, labels).item()
                acc_values.append(acc)
        return np.mean(acc_values)


    # list where all the routing point got appended to
    routing_point_list = []

    parser = configargparse.get_argument_parser()
    parser = toy_args(parser)
    conf = parser.parse_args()

    # get logger and log config
    log = get_logger(__name__)
    log.info(parser.format_values())

    train_data = Gaussian2D(transform=lambda x: torch.from_numpy(x).type(torch.FloatTensor), n_samples=(2000, 2000),
                            train=True, new_dim=conf.data_dim)
    val_data = Gaussian2D(transform=lambda x: torch.from_numpy(x).type(torch.FloatTensor), n_samples=(2000, 2000),
                          train=False, new_dim=conf.data_dim)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_data, batch_size=128, drop_last=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=128, drop_last=True, **kwargs)

    model = ToyCapsNet(in_features=conf.data_dim, final_caps=2, final_len=2, prim_caps=20, prim_len=2,
                       routing_iters=conf.routing_iters, bias_routing=conf.bias_routing, recon=conf.use_recon,
                       sparse=conf.sparse)

    if torch.cuda.is_available():
        model.cuda()

    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, 0, include_recon=conf.use_recon,
                               include_entropy=False, caps_sizes=[2])

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(conf.epochs):

        for batch_idx, batch in enumerate(train_loader):

            acc = train(batch, epoch, batch_idx)
        log.info("\nEpoch: {}, acc: {:0.4f}".format(epoch, acc))

    log.info("Final acc: {:0.4f}".format(evaluate()))

    write_folder = f"experiments/{conf.output_folder}"
    write_file = f"{write_folder}/{conf.output_file}"

    if not os.path.exists(write_folder):
        if not os.path.exists("experiments"):
            raise NotADirectoryError("experiments folder does not exists")
        os.makedirs(write_folder)

    with open(write_file, 'wb') as f:
        log.info(f"Write routing point list to {write_file}")
        pickle.dump(routing_point_list, f)


if __name__ == "__main__":
    main()
