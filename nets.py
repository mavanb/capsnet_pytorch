import torch
from torch import nn
from layers import Conv2dPrimaryLayer, DenseCapsuleLayer, LinearPrimaryLayer, DynamicRouting
from utils import one_hot, new_grid_size, padding_same_tf, init_weights, flex_profile, get_device
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
        # if labels is None:
        targets = self.compute_predictions(self.compute_logits(final_caps))
        # else:
        #     targets = labels

        masks = one_hot(targets, self.num_final_caps)
        masked_caps = final_caps * masks[:, :, None]
        decoder_input = masked_caps.view(final_caps.shape[0], -1)
        return decoder_input


class ToyCapsNet(_CapsNet):
    """ Toy Capsule Network to classify

    The primary capsules are constructed using one linear layer without (no non-linearity) and the squash function. The
    network has one dense capsule layer.

    The decoder network uses two linear layers with a non linearity in between.

    Args:
        in_features (int): Number of dims of the input vector.
        final_caps (int): Number of capsule in the output / Number of classes.
        final_len (int): Vector length of final capsules.
        prim_caps (int): Number of primary capsules.
        prim_len (int): Vector length of the primary capsules.
        routing_iters (int): Number of routing interations.
        bias_routing (bool): Add bias to routing yes/no.


    """

    def __init__(self, in_features, final_caps, final_len, prim_caps, prim_len, routing_iters, bias_routing, recon, sparse):
        super().__init__(final_caps)

        self.routing_iters = routing_iters

        self.primary_caps_layer = LinearPrimaryLayer(in_features, prim_caps, prim_len)
        self.dense_caps_layer = DenseCapsuleLayer(j=final_caps, i=prim_caps, m=prim_len, n=final_len, stdev=0.1)
        self.rout_layer = DynamicRouting(j=final_caps, n=final_len, bias_routing=bias_routing, sparse=sparse)

        self.caps_sizes = torch.tensor([final_caps], device=get_device(),
                                       requires_grad=False)

        self.recon = recon

        self.decoder = nn.Sequential(
            nn.Linear(final_len * final_caps, 52),
            nn.ReLU(inplace=True),
            nn.Linear(52, in_features),
        )

    def forward(self, x, t=None):

        # compute grid of capsules
        primary_caps = self.primary_caps_layer(x)

        # compute for each child a parent prediction
        all_caps = self.dense_caps_layer(primary_caps)

        final_caps, _ = self.rout_layer(all_caps, self.routing_iters)

        # compute the logits by taking the norm
        logits = self.compute_logits(final_caps)

        # flatten final caps and mask all but target if known
        decoder_input = self.create_decoder_input(final_caps, t)

        # create reconstruction
        if self.recon:
            recon = self.decoder(decoder_input)
        else:
            recon = None

        return logits, recon, final_caps


class BasicCapsNet(_CapsNet):
    """ Basic capsule network with dynamic architecture.

    Args:
        in_channels (int): Number of channels of the input imaga.
        routing_iters (int): Number of iterations of the routing algo.
        in_height (int): Height of the input image.
        in_width (int): Width of the input image.
        stdev_W (float):  Value of the weight initialization of the dense capsule layers.
        bias_routing (bool): Add bias to routing yes/no.
        arch (obj Architecture): Architecture of the capsule network.
        recon (bool): Use reconstruction yes/no.
        sparse (str):  Sparse method, see docs or config for formatting convention.
    """

    def __init__(self, in_channels, routing_iters, in_height, in_width, stdev_W, bias_routing,
                 arch, recon, sparse):

        # init parent CapsNet class with number of capsule in the final layer
        super().__init__(arch.final.caps)

        self.arch = arch
        self.routing_iters = routing_iters
        self.recon = recon

        # get capsule sizes from the architecture (arch) and cast to tensor
        self.caps_sizes = torch.tensor([l.caps for l in arch.all_but_prim], device=get_device(),
                                       requires_grad=False)

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
        dense_layers = torch.nn.ModuleList()
        rout_layers = torch.nn.ModuleList()

        # set input of first layer to the primary layer
        in_caps = in_features_dense_layer
        in_len = arch.prim.len

        # loop over all other layers
        for h in arch.all_but_prim:

            # set capsules number and length to the current layer output
            out_caps = h.caps
            out_len = h.len

            dense_layer = DenseCapsuleLayer(j=out_caps, i=in_caps, m=in_len, n=out_len, stdev=stdev_W)

            rout_layer = DynamicRouting(j=out_caps, n=out_len, bias_routing=bias_routing, sparse=sparse)

            # add all in right order to layer list
            dense_layers.append(dense_layer)
            rout_layers.append(rout_layer)

            # capsules number and length to the next layer input
            in_caps = out_caps
            in_len = out_len

        self.dense_layers = dense_layers
        self.rout_layers = rout_layers

        if recon:
            self.decoder = CapsNetDecoder(arch.final.len, arch.final.caps, in_channels, in_height, in_width)

    def set_sparse_on(self):
        """ Set sparsify back on in all rout layers in this class."""
        for rout_layer in self.rout_layers:
            rout_layer.sparse.set_on()

    def set_sparse_off(self):
        """ Set sparsify back off in all rout layers in this class."""
        for rout_layer in self.rout_layers:
            rout_layer.sparse.set_off()

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
        # entropy_list = []
        entropy = torch.zeros(len(self.dense_layers), b, self.routing_iters, device=get_device(), requires_grad=False)

        # loop over the capsule layers
        for layer_idx, (dense_layer, rout_layer) in enumerate(zip(self.dense_layers, self.rout_layers)):

            # compute for each child a parent prediction
            all_caps = dense_layer(caps_input)

            # take weighted average of parent prediction, weights determined based on correspondence of predictions
            caps_input, entropy_layer = rout_layer(all_caps, self.routing_iters)

            # entropy_list.append(entropy_stats)
            entropy[layer_idx, :, :] = entropy_layer

        # final capsule are the output of last layer
        final_caps = caps_input

        # compute the logits by taking the norm
        logits = self.compute_logits(final_caps)

        # flatten final caps and mask all but target if known
        decoder_input = self.create_decoder_input(final_caps, t)

        # create reconstruction
        if self.recon:
            recon = self.decoder(decoder_input)
        else: 
            recon = None

        return logits, recon, final_caps, entropy


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


