import torch
from torch.autograd import Variable
from torch import nn
from layers import Conv2dPrimaryLayer, DenseCapsuleLayer, LinearPrimaryLayer, DynamicRouting
from utils import one_hot, new_grid_size, padding_same_tf, init_weights, flex_profile
import torch.nn.functional as F
from torch.nn.modules.module import _addindent
import numpy as np


class _Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.epoch = 0

    @staticmethod
    def compute_predictions(logits):
        """Compute predictions b    y selecting
        :param logits: [batch_size, num_classes/capsules]
        :returns [batch_size]
        """
        _, index_max = logits.max(dim=1, keepdim=False)
        return index_max.squeeze()

    def compute_acc(self, logits, label):
        """ Compute accuracy of batch
        :param logits: [batch_size, classes]
        :param label: [batch_size]
        :return: batch accurarcy (float)
        """
        return sum(self.compute_predictions(logits) == label).float() / logits.size(0)

    @staticmethod
    def _num_parameters(module):
        return int(np.sum([np.prod(list(p.shape)) for p in module.parameters()]))

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  ({}, {}): '.format(key, self._num_parameters(module)) + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr


class _CapsNet(_Net):
    """ Abstract CapsuleNet class."""

    def __init__(self, num_final_caps):
        super().__init__()
        self.num_final_caps = num_final_caps

    @staticmethod
    def compute_logits(caps):
        """ Compute class logits from the capsule/vector length.
        Norm is safe, to avoid nan's.

        :param caps: capsules of shape [batch_size, num_capsules, dim_capsules]
        :returns logits of shape [batch_size, num_capsules]
        """
        return torch.sqrt((caps ** 2).sum(dim=-1, keepdim=False) + 1e-9)

    @staticmethod
    def compute_probs(logits):
        """ Compute the class probabilities from the logits using the softmax

        Args:
            logits: (tensor) logits of final capsules of shape [batch_size, num_classes]

        Returns: (tensor) the corresponding class probabilities of shape [batch_size, num_classes]
        """
        return F.softmax(logits, dim=1)

    def create_decoder_input(self, final_caps, labels=None):
        """ Construct decoder input based on class probs and final capsules.
        Flattens capsules to [batch_size, num_final_caps * dim_final_caps] and sets all values which do not come from
        the correct class/capsule to zero (masks). During training the labels are used to masks, during inference the
        max of the class probabilities.
        :param labels: [batch_size, 1], if None: use predictions
        :return: [batch_size, num_final_caps * dim_final_caps]
        """
        targets = labels if type(labels) == Variable else self.compute_predictions(self.compute_logits(final_caps))
        if type(labels) == Variable:
            pass
        masks = one_hot(targets, self.num_final_caps)
        masked_caps = final_caps * masks[:, :, None]
        decoder_input = masked_caps.view(final_caps.shape[0], -1)
        return decoder_input


class ToyCapsNet(_CapsNet):

    def __init__(self, in_features, final_caps, vec_len_prim, vec_len_final, routing_iters, prim_caps, bias_routing):
        super().__init__(final_caps)
        self.routing_iters = routing_iters
        self.primary_caps_layer = LinearPrimaryLayer(in_features, prim_caps, vec_len_prim)
        self.dense_caps_layer = DenseCapsuleLayer(prim_caps, final_caps, vec_len_prim,
                                                  vec_len_final, routing_iters, stdev=0.1)

        self.dynamic_routing = None

        self.decoder = nn.Sequential(
            nn.Linear(vec_len_final * final_caps, 52),
            nn.ReLU(inplace=True),
            nn.Linear(52, in_features),
        )

    def forward(self, x, t=None):

        # compute grid of capsules
        primary_caps = self.primary_caps_layer(x)

        # for each capsule in primary layer compute prediction for all next layer capsules
        all_final_caps = self.dense_caps_layer(primary_caps)

        # compute digit capsules
        final_caps, routing_point = self.dynamic_routing(all_final_caps, self.routing_iters, self.b_routing)
        logits = self.compute_logits(final_caps)
        decoder_input = self.create_decoder_input(final_caps, t)
        recon = self.decoder(decoder_input)

        return logits, recon, final_caps, routing_point


class BasicCapsNet(_CapsNet):

    def __init__(self, in_channels, routing_iters, in_height, in_width, stdev_W, bias_routing,
                 sparse_threshold, sparsify, sparse_topk, arch):
        super().__init__(10) #todo remove, retrieve from data

        self.arch = arch
        self.routing_iters = routing_iters

        prim_caps = arch.prim.caps
        prim_len = arch.prim.len

        # initial convolution
        conv_channels = prim_caps * prim_len
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=9, stride=1, padding=0,
                          bias=True)
        self.conv1 = init_weights(conv1)
        self.relu = nn.ReLU()

        # compute primary capsules
        self.primary_caps_layer = Conv2dPrimaryLayer(in_channels=conv_channels, out_channels=prim_caps,
                                                     vec_len=prim_len)

        # grid of multiple primary caps channels is flattend, number of new channels: grid * point * channels in grid
        new_height, new_width = new_grid_size(new_grid_size((in_height, in_width), kernel_size=9), kernel_size=9, stride=2)
        in_features_dense_layer = new_height * new_width * prim_caps

        # init list for all hidden parts
        # dense_layers = []
        # rout_layers = []
        dense_layers = torch.nn.ModuleList()
        rout_layers = torch.nn.ModuleList()

        # set input of first layer to the primary layer
        in_caps = in_features_dense_layer
        in_len = arch.prim.len

        # loop over all other layers
        for h in arch.other_layers:

            # set capsules number and length to the current layer output
            out_caps = h.caps
            out_len = h.len

            dense_layer = DenseCapsuleLayer(j=out_caps, i=in_caps, m=in_len, n=out_len, stdev=stdev_W)

            rout_layer = DynamicRouting(j=out_caps, n=out_len, bias_routing=bias_routing,
                                        sparse_threshold=sparse_threshold, sparsify=sparsify, sparse_topk=sparse_topk)

            # add all in right order to layer list
            dense_layers.append(dense_layer)
            rout_layers.append(rout_layer)

            # capsules number and length to the next layer input
            in_caps = out_caps
            in_len = out_len

        self.dense_layers = dense_layers
        self.rout_layers = rout_layers

        self.decoder = CapsNetDecoder(arch.final.len, arch.final.caps, in_channels, in_height, in_width)

    def set_sparsify(self, value):
        """ Set sparsify. Can, for example, be used to turn sparsify off during inference."""
        for rout_layer in self.rout_layers:
            rout_layer.sparsify = value

    @flex_profile
    def forward(self, x, t=None):

        # apply conv layer
        conv1 = self.relu(self.conv1(x))

        # compute grid of capsules
        primary_caps = self.primary_caps_layer(conv1)

        # flatten primary capsules
        b, c, w, h, m = primary_caps.shape
        primary_caps_flat = primary_caps.view(b, c * w * h, m)

        # set initial input of capsule part of the network
        caps_input = primary_caps_flat

        # list for all routing stats
        entropy_list = []

        # loop over the capsule layers
        for dense_layer, rout_layer in zip(self.dense_layers, self.rout_layers):

            # compute for each child a parent prediction
            all_caps = dense_layer(caps_input)

            # take weighted average of parent prediction, weights determined based on correspondence of predictions
            caps_input, entropy_stats = rout_layer(all_caps, self.routing_iters)

            entropy_list.append(entropy_stats)

        # final capsule are the output of last layer
        final_caps = caps_input

        # compute the logits by taking the norm
        logits = self.compute_logits(final_caps)

        # flatten final caps and mask all but target if known
        decoder_input = self.create_decoder_input(final_caps, t)

        # create reconstruction
        recon = self.decoder(decoder_input)

        return logits, recon, final_caps, entropy_list


class BaselineCNN(_Net):
    """
    Convnet as implemented in https://github.com/Sarasra/models/blob/master/research/capsules/models/conv_model.py.
    The paper (Hinton 2017) mentions a slightly different architecture than in the source code.
    """
    def __init__(self, clases, in_channels, in_height, in_width):
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=5)
        grid1_conv = new_grid_size((in_height, in_width), kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=padding_same_tf(grid1_conv, 2, 2))
        grid1_pool = new_grid_size(grid1_conv, kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=5)
        grid2_conv = new_grid_size(grid1_pool, kernel_size=5)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=padding_same_tf(grid2_conv, 2, 2))
        grid2_pool = new_grid_size(grid2_conv, kernel_size=2, stride=2)
        self.grid2_flat = grid2_pool[0] * grid2_pool[1] * 256

        self.fc1 = nn.Linear(self.grid2_flat, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.fc3 = nn.Linear(192, classes)

        self.epoch = 0

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.max_pool1(x))
        x = self.conv2(x)
        x = F.relu(self.max_pool2(x))
        x = x.view(-1, self.grid2_flat)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CapsNetDecoder(nn.Module):

    def __init__(self, vec_len_digit, digit_caps, in_channels, in_height, in_width):
        super(CapsNetDecoder, self).__init__()

        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width

        self.flat_reconstruction = nn.Sequential(
            init_weights(nn.Linear(vec_len_digit * digit_caps, 512)),
            nn.ReLU(inplace=True),
            init_weights(nn.Linear(512, 1024)),
            nn.ReLU(inplace=True),
            init_weights(nn.Linear(1024, in_channels * in_height * in_width)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flat_reconstruction(x).view(-1, self.in_channels, self.in_height, self.in_width)
        return x


