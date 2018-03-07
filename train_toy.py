from __future__ import print_function

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from configurations.config_utils import get_conf_logger
from data.data_loader import Gaussian2D
from loss import CapsuleLoss
from nets import ToyCapsNet
from utils import squash
from utils import variable


def custom_args(parser):
    parser.add('--toy_capsnet_config', is_config_file=True, default="configurations/toy_capsnet.conf",
               help='configurations file path')
    parser.add_argument('--model_name', type=str, default="toy_caps_net", help='Name of the model.')
    parser.add_argument('--alpha', type=float, required=True, help="Alpha of CapsuleLoss")
    parser.add_argument('--m_plus', type=float, required=True, help="m_plus of margin loss")
    parser.add_argument('--m_min', type=float, required=True, help="m_min of margin loss")
    parser.add_argument('--prim_caps', type=int, required=True, help="Number of primary capsules")
    parser.add_argument('--routing_iters', type=int, required=True, help="Number of iterations in the routing algo.")
    parser.add_argument('--bias_routing', type=bool, default=False, help="whether to use bias in routing")
    return parser


def logged_dynamic_routing(u_hat, iters, bias):
    """
    Implementation of routing algorithm described in Dynamic Routing Hinton 2017.
    :param input: u_hat, Variable containing the of the next layer capsules given each previous layer capsule. shape:
    [batch_size, num_caps_next_layer, num_caps_prev_layer, dim_next_layer]
    :param iters: number of iterations in routing algo.
    :return: next layer predictions sized by probability/correspondence. shape: [batch_size, num_caps_next_layer,
    dim_next_layer]
    """
    routing_point = RoutingPoint(u_hat.data.cpu().numpy() if torch.cuda.is_available() else u_hat.data.numpy())

    b, j, i, n = u_hat.shape
    b_vec = variable(torch.zeros(b, j, i))

    for index in range(iters):

        # softmax of i, weight of all predictions should sum to 1, note in tf code this does not give an error
        c_vec = torch.nn.Softmax(dim=1)(b_vec)

        # in einsum: bij, bjin-> bjn
        # in matmul: bj1i, bjin = bj (1i)(in) -> bjn
        s_vec = torch.matmul(c_vec.view(b, j, 1, i), u_hat).squeeze()
        if type(bias) == torch.nn.Parameter:
            s_vec_bias = s_vec + bias
        else:
            s_vec_bias = s_vec
        v_vec = squash(s_vec_bias)

        if index < (iters - 1):  # skip update last iter
            # in einsum: "bjin, bjn-> bij", inner product over n
            # in matmul: bji1n, bj1n1 = bji (1n)(n1) = bji1
            # note: use x=x+1 instead of x+=1 to ensure new object creation and avoid inplace operation
            b_vec = b_vec + torch.matmul(u_hat.view(b, j, i, 1, n), v_vec.view(b, j, 1, n, 1)).squeeze()

        if torch.cuda.is_available():
            routing_iter = RoutingIter(index, c_vec.data.cpu().numpy(), s_vec.data.cpu().numpy(),
                                       v_vec.data.cpu().numpy())
        else:
            routing_iter = RoutingIter(index, c_vec.data.numpy(), s_vec.data.numpy(),
                                       v_vec.data.numpy())
        routing_point.routing_iters.append(routing_iter)

    return v_vec, routing_point


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
    conf, logger = get_conf_logger(custom_args=custom_args)

    new_dim = 3
    train_data = Gaussian2D(transform=lambda x: torch.from_numpy(x).type(torch.FloatTensor), n_samples=(2000, 2000), train=True, new_dim=new_dim)
    val_data = Gaussian2D(transform=lambda x: torch.from_numpy(x).type(torch.FloatTensor), n_samples=(2000, 2000), train=False, new_dim=new_dim)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_data, batch_size=128, drop_last=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=128, drop_last=True, **kwargs)

    model = ToyCapsNet(in_features=new_dim, final_caps=2, vec_len_prim=2, routing_iters=conf.routing_iters,
                       prim_caps=20, vec_len_final=2, bias_routing=conf.bias_routing)

    if torch.cuda.is_available():
        model.cuda()

    model.dynamic_routing = logged_dynamic_routing

    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, num_classes=2)

    optimizer = torch.optim.Adam(model.parameters())

    routing_point_list = []

    def train(batch, epoch, iter):
        observed_batch, label_batch, latent_batch = batch
        model.train()
        optimizer.zero_grad()

        data = variable(observed_batch)
        labels = variable(label_batch)

        class_logits, reconstruction, _, routing_point = model(data, labels)

        # add all training step info to the routing point
        acc = evaluate()
        routing_point.acc = acc
        routing_point.iter = iter
        routing_point.epoch = epoch
        routing_point.latent = latent_batch
        routing_point.label = label_batch
        routing_point_list.append(routing_point)

        loss, margin_loss, recon_loss = capsule_loss(data, labels, class_logits, reconstruction)

        loss.backward()
        optimizer.step()
        return acc

    def evaluate():
        model.eval()
        acc_values = []
        for batch_idx, batch in enumerate(val_loader):
            data = variable(batch[0], volatile=True)
            labels = variable(batch[1])
            class_probs, _, _, _ = model(data)
            acc = model.compute_acc(class_probs, labels)
            acc_values.append(acc)
        return np.mean(acc_values)

    for epoch in range(conf.epochs):

        for batch_idx, batch in enumerate(train_loader):

            acc = train(batch, epoch, batch_idx)
        logger("Epoch: {}, acc: {:0.2f}".format(epoch, acc))

    logger("Final acc: {}".format(evaluate()))

    routing_file_path = conf.model_checkpoint_path
    if not os.path.exists(conf.trained_model_path):
        os.makedirs(conf.trained_model_path)

    with open(routing_file_path, 'wb') as f:
        pickle.dump(routing_point_list, f)


if __name__ == "__main__":
    main()
