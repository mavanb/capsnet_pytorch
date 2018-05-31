from __future__ import print_function

from torch import nn
from layers import Conv2dPrimaryLayer, DenseCapsuleLayer, DynamicRouting
from nets import _CapsNet, CapsNetDecoder
from utils import new_grid_size, init_weights

import torch
from torchvision import transforms

from configurations.general_confs import parse_bool, get_conf
from data.data_loader import get_dataset
from ignite_features.trainer import CapsuleTrainer
from loss import CapsuleLoss
from utils import get_logger
from train_capsnet import custom_args


class DoubleCapsNet(_CapsNet):

    def __init__(self, in_channels, digit_caps, vec_len_prim, vec_len_digit, routing_iters, prim_caps, in_height,
                 in_width, softmax_dim, squash_dim, stdev_W, bias_routing, sparse_threshold, sparsify, sparse_topk):
        super().__init__(digit_caps)

        self.routing_iters = routing_iters

        # initial convolution
        conv_channels = prim_caps * vec_len_prim
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=9, stride=1, padding=0,
                          bias=True)
        self.conv1 = init_weights(conv1)
        self.relu = nn.ReLU()

        # compute primary capsules
        self.primary_caps_layer = Conv2dPrimaryLayer(in_channels=conv_channels, out_channels=prim_caps,
                                                     vec_len=vec_len_prim, squash_dim=squash_dim)

        # grid of multiple primary caps channels is flattend, number of new channels: grid * point * channels in grid
        new_height, new_width = new_grid_size(new_grid_size((in_height, in_width), kernel_size=9), 9, 2)
        in_features_dense_layer = new_height * new_width * prim_caps

        self.dense_caps_layer1 = DenseCapsuleLayer(in_capsules=in_features_dense_layer, out_capsules=100,
                                                   vec_len_in=vec_len_prim, vec_len_out=12, routing_iters=routing_iters,
                                                   stdev=stdev_W)

        self.dynamic_routing1 = DynamicRouting(j=100, i=in_features_dense_layer, n=12, softmax_dim=softmax_dim,
                                               bias_routing=bias_routing, sparse_threshold=sparse_threshold,
                                               sparsify=sparsify)

        self.dense_caps_layer2 = DenseCapsuleLayer(in_capsules=100, out_capsules=digit_caps,
                                                   vec_len_in=12, vec_len_out=vec_len_digit,
                                                   routing_iters=routing_iters,
                                                   stdev=stdev_W)

        self.dynamic_routing2 = DynamicRouting(j=digit_caps, i=100, n=vec_len_digit, softmax_dim=softmax_dim,
                                               bias_routing=bias_routing, sparse_threshold=sparse_threshold,
                                               sparsify=False)

        self.decoder = CapsNetDecoder(vec_len_digit, digit_caps, in_channels, in_height, in_width)

        self.softmax_dim = softmax_dim

    def set_sparsify(self, value):
        """ Set sparsify. Can, for example, be used to turn sparsify off during inference."""
        self.dynamic_routing1.sparsify = value

    def forward(self, x, t=None):
        # apply conv layer
        conv1 = self.relu(self.conv1(x))

        # compute grid of capsules
        primary_caps = self.primary_caps_layer(conv1)

        b, c, w, h, m = primary_caps.shape
        primary_caps_flat = primary_caps.view(b, c * w * h, m)

        ### test extra layer
        all_caps1 = self.dense_caps_layer1(primary_caps_flat)
        caps1, stats = self.dynamic_routing1(all_caps1, self.routing_iters)
        all_caps2 = self.dense_caps_layer2(caps1)
        final_caps, _ = self.dynamic_routing2(all_caps2, self.routing_iters)

        logits = self.compute_logits(final_caps)

        decoder_input = self.create_decoder_input(final_caps, t)
        recon = self.decoder(decoder_input)

        return logits, recon, final_caps, stats


class TripleCapsNet(_CapsNet):

    def __init__(self, in_channels, digit_caps, vec_len_prim, vec_len_digit, routing_iters, prim_caps, in_height,
                 in_width, softmax_dim, squash_dim, stdev_W, bias_routing, sparse_threshold, sparsify, sparse_topk):
        super().__init__(digit_caps)

        self.routing_iters = routing_iters

        # initial convolution
        conv_channels = prim_caps * vec_len_prim
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=9, stride=1, padding=0,
                          bias=True)
        self.conv1 = init_weights(conv1)
        self.relu = nn.ReLU()

        # compute primary capsules
        self.primary_caps_layer = Conv2dPrimaryLayer(in_channels=conv_channels, out_channels=prim_caps,
                                                     vec_len=vec_len_prim, squash_dim=squash_dim)

        # grid of multiple primary caps channels is flattend, number of new channels: grid * point * channels in grid
        new_height, new_width = new_grid_size(new_grid_size((in_height, in_width), kernel_size=9), 9, 2)
        in_features_dense_layer = new_height * new_width * prim_caps

        self.dense_caps_layer1 = DenseCapsuleLayer(in_capsules=in_features_dense_layer, out_capsules=100,
                                                   vec_len_in=vec_len_prim, vec_len_out=12, routing_iters=routing_iters,
                                                   stdev=stdev_W)

        self.dynamic_routing1 = DynamicRouting(j=100, i=in_features_dense_layer, n=12, softmax_dim=softmax_dim,
                                               bias_routing=bias_routing, sparse_threshold=sparse_threshold,
                                               sparsify=sparsify)

        self.dense_caps_layer2 = DenseCapsuleLayer(in_capsules=100, out_capsules=50,
                                                   vec_len_in=12, vec_len_out=11,
                                                   routing_iters=routing_iters,
                                                   stdev=stdev_W)

        self.dynamic_routing2 = DynamicRouting(j=50, i=100, n=11, softmax_dim=softmax_dim,
                                               bias_routing=bias_routing, sparse_threshold=sparse_threshold,
                                               sparsify=sparsify)

        self.dense_caps_layer3 = DenseCapsuleLayer(in_capsules=50, out_capsules=digit_caps,
                                                   vec_len_in=11, vec_len_out=vec_len_digit,
                                                   routing_iters=routing_iters,
                                                   stdev=stdev_W)

        self.dynamic_routing3 = DynamicRouting(j=digit_caps, i=50, n=vec_len_digit, softmax_dim=softmax_dim,
                                               bias_routing=bias_routing, sparse_threshold=sparse_threshold,
                                               sparsify=sparsify)

        self.decoder = CapsNetDecoder(vec_len_digit, digit_caps, in_channels, in_height, in_width)

        self.softmax_dim = softmax_dim

    def set_sparsify(self, value):
        """ Set sparsify. Can, for example, be used to turn sparsify off during inference."""
        self.dynamic_routing1.sparsify = value
        self.dynamic_routing2.sparsify = value

    def forward(self, x, t=None):
        # apply conv layer
        conv1 = self.relu(self.conv1(x))

        # compute grid of capsules
        primary_caps = self.primary_caps_layer(conv1)

        b, c, w, h, m = primary_caps.shape
        primary_caps_flat = primary_caps.view(b, c * w * h, m)

        ### test extra layer
        all_caps1 = self.dense_caps_layer1(primary_caps_flat)
        caps1, _ = self.dynamic_routing1(all_caps1, self.routing_iters)
        all_caps2 = self.dense_caps_layer2(caps1)
        caps2, _ = self.dynamic_routing2(all_caps2, self.routing_iters)
        all_caps3 = self.dense_caps_layer3(caps2)
        final_caps, stats3 = self.dynamic_routing3(all_caps3, self.routing_iters)

        stats = stats3

        logits = self.compute_logits(final_caps)

        decoder_input = self.create_decoder_input(final_caps, t)
        recon = self.decoder(decoder_input)

        return logits, recon, final_caps, stats



def main():
    conf, parser = get_conf(custom_args)
    log = get_logger(__name__)
    log.info(parser.format_values())
    transform = transforms.ToTensor()
    dataset, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)

    model = TripleCapsNet(in_channels=data_shape[0], digit_caps=label_shape, vec_len_prim=8, vec_len_digit=16,
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
