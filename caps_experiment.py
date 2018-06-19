from __future__ import print_function

from torch import nn
from layers import Conv2dPrimaryLayer, DenseCapsuleLayer, DynamicRouting
from nets import _CapsNet, CapsNetDecoder
from utils import new_grid_size, init_weights

import torch
from torchvision import transforms

from configurations.general_confs import get_conf, capsule_arguments
from data.data_loader import get_dataset
from ignite_features.trainer import CapsuleTrainer
from loss import CapsuleLoss
from utils import get_logger, flex_profile


class DoubleCapsNet(_CapsNet):

    def __init__(self, in_channels, routing_iters, in_height, in_width, stdev_W, bias_routing,
                 sparse_threshold, sparsify, sparse_topk, arch):
        super().__init__(10) #todo remove, retrieve from data

        prim_caps = arch.prim.caps
        prim_len = arch.prim.len

        self.routing_iters = routing_iters

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
        layers = []

        # set input of first layer to the primary layer
        in_caps = in_features_dense_layer
        in_len = arch.prim.len

        # loop over all other layers
        for h in arch.layers:

            # set capsules number and length to the current layer output
            out_caps = h.caps
            out_len = h.len

            dense_layer = DenseCapsuleLayer(i=in_caps, j=out_caps, m=in_len, n=out_len, stdev=stdev_W)

            rout_layer = DynamicRouting(j=out_caps, n=out_len, bias_routing=bias_routing,
                                        sparse_threshold=sparse_threshold, sparsify=sparsify, sparse_topk=sparse_topk)

            # add all in right order to layer list
            layers.append(dense_layer)
            layers.append(rout_layer)

            # capsules number and length to the next layer input
            in_caps = out_caps
            in_len = out_len

        # self.caps_part = nn.Sequential(
        #     *layers,
        # ) #todo, check how to solve stats return problm
        self.caps_part = layers

        self.decoder = CapsNetDecoder(arch.final.len, arch.final.caps, in_channels, in_height, in_width)


    def set_sparsify(self, value):
        """ Set sparsify. Can, for example, be used to turn sparsify off during inference."""
        self.dynamic_routing1.sparsify = value
        #todo fix this for all routing layers

    @flex_profile
    def forward(self, x, t=None):
        # apply conv layer
        conv1 = self.relu(self.conv1(x))

        # compute grid of capsules
        primary_caps = self.primary_caps_layer(conv1)

        b, c, w, h, m = primary_caps.shape
        primary_caps_flat = primary_caps.view(b, c * w * h, m)

        all_caps1 = self.dense_caps_layer1(primary_caps_flat)
        caps1, stats1 = self.dynamic_routing1(all_caps1, self.routing_iters)

        all_caps2 = self.dense_caps_layer2(caps1)
        final_caps, stats2 = self.dynamic_routing2(all_caps2, self.routing_iters)

        logits = self.compute_logits(final_caps)

        decoder_input = self.create_decoder_input(final_caps, t)
        recon = self.decoder(decoder_input)

        # assert stats1["H_c_vec"], "Routing stats should contain H_c_vec"
        # assert stats2["H_c_vec"], "Routing stats should contain H_c_vec"
        # assert stats1["H_c_vec"].keys() == stats2["H_c_vec"].keys(), "Assumes that both dict have the same number of " \
        #                                                              "routing iterations."

        stats = {}
        stats["H_c_vec"] = {}
        for routing_key in stats1["H_c_vec"].keys():
            stats["H_c_vec"][routing_key] = (stats1["H_c_vec"][routing_key] + stats2["H_c_vec"][routing_key])/2

        return logits, recon, final_caps, stats


class LongCapsNet(_CapsNet):

    def __init__(self, in_channels, digit_caps, vec_len_prim, vec_len_digit, routing_iters, prim_caps, in_height,
                 in_width, softmax_dim, squash_dim, stdev_W, bias_routing, sparse_threshold, sparsify, sparse_topk, hidden_capsules):
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

        self.dense_caps_layer1 = DenseCapsuleLayer(i=in_features_dense_layer, j=hidden_capsules,
                                                   m=vec_len_prim, n=vec_len_digit, stdev=stdev_W)

        self.dynamic_routing1 = DynamicRouting(j=hidden_capsules, i=in_features_dense_layer, n=vec_len_digit, softmax_dim=softmax_dim,
                                               bias_routing=bias_routing, sparse_threshold=sparse_threshold,
                                               sparsify=sparsify, sparse_topk=sparse_topk)

        self.dense_caps_layer2 = DenseCapsuleLayer(i=hidden_capsules, j=hidden_capsules,
                                                   m=vec_len_digit, n=vec_len_digit,
                                                   stdev=stdev_W)

        self.dynamic_routing2 = DynamicRouting(j=hidden_capsules, i=hidden_capsules, n=vec_len_digit, softmax_dim=softmax_dim,
                                               bias_routing=bias_routing, sparse_threshold=sparse_threshold,
                                               sparsify=sparsify, sparse_topk=sparse_topk)

        # self.dense_caps_layer3 = DenseCapsuleLayer(in_capsules=hidden_capsules, out_capsules=hidden_capsules,
        #                                            vec_len_in=vec_len_digit, vec_len_out=vec_len_digit,
        #                                            routing_iters=routing_iters,
        #                                            stdev=stdev_W)
        #
        # self.dynamic_routing3 = DynamicRouting(j=hidden_capsules, i=hidden_capsules, n=vec_len_digit, softmax_dim=softmax_dim,
        #                                        bias_routing=bias_routing, sparse_threshold=sparse_threshold,
        #                                        sparsify=sparsify, sparse_topk=sparse_topk)

        # self.dense_caps_layer4 = DenseCapsuleLayer(in_capsules=hidden_capsules, out_capsules=hidden_capsules,
        #                                            vec_len_in=vec_len_digit, vec_len_out=vec_len_digit,
        #                                            routing_iters=routing_iters,
        #                                            stdev=stdev_W)
        #
        # self.dynamic_routing4 = DynamicRouting(j=hidden_capsules, i=hidden_capsules, n=vec_len_digit, softmax_dim=softmax_dim,
        #                                        bias_routing=bias_routing, sparse_threshold=sparse_threshold,
        #                                        sparsify=sparsify, sparse_topk=sparse_topk)
        #
        # self.dense_caps_layer5 = DenseCapsuleLayer(in_capsules=hidden_capsules, out_capsules=hidden_capsules,
        #                                            vec_len_in=vec_len_digit, vec_len_out=vec_len_digit,
        #                                            routing_iters=routing_iters,
        #                                            stdev=stdev_W)
        #
        # self.dynamic_routing5 = DynamicRouting(j=hidden_capsules, i=hidden_capsules, n=vec_len_digit, softmax_dim=softmax_dim,
        #                                        bias_routing=bias_routing, sparse_threshold=sparse_threshold,
        #                                        sparsify=sparsify, sparse_topk=sparse_topk)

        self.dense_caps_layer6 = DenseCapsuleLayer(i=hidden_capsules, j=digit_caps,
                                                   m=vec_len_digit, n=vec_len_digit,
                                                   stdev=stdev_W)

        self.dynamic_routing6 = DynamicRouting(j=digit_caps, i=hidden_capsules, n=vec_len_digit, softmax_dim=softmax_dim,
                                               bias_routing=bias_routing, sparse_threshold=sparse_threshold,
                                               sparsify=sparsify, sparse_topk=sparse_topk)

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

        ### tests extra layer
        all_caps = self.dense_caps_layer1(primary_caps_flat)
        caps, _ = self.dynamic_routing1(all_caps, self.routing_iters)

        # all_caps = self.dense_caps_layer2(caps)
        # caps, _ = self.dynamic_routing2(all_caps, self.routing_iters)

        # all_caps = self.dense_caps_layer3(caps)
        # caps, _ = self.dynamic_routing3(all_caps, self.routing_iters)

        # all_caps = self.dense_caps_layer4(caps)
        # caps, _ = self.dynamic_routing4(all_caps, self.routing_iters)
        #
        # all_caps = self.dense_caps_layer5(caps)
        # caps, _ = self.dynamic_routing5(all_caps, self.routing_iters)

        all_caps = self.dense_caps_layer6(caps)
        final_caps, stats = self.dynamic_routing6(all_caps, self.routing_iters)

        logits = self.compute_logits(final_caps)

        decoder_input = self.create_decoder_input(final_caps, t)
        recon = self.decoder(decoder_input)

        return logits, recon, final_caps, stats


def main():
    custom_args = capsule_arguments("exp_capsnet")
    conf, parser = get_conf(custom_args)
    log = get_logger(__name__)
    log.info(parser.format_values())
    transform = transforms.ToTensor()
    data_train, data_test, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)

    model = DoubleCapsNet(in_channels=data_shape[0], digit_caps=label_shape, vec_len_prim=8, vec_len_digit=16,
                        routing_iters=conf.routing_iters, prim_caps=conf.prim_caps, in_height=data_shape[1],
                        in_width=data_shape[2], softmax_dim=conf.softmax_dim, squash_dim=conf.squash_dim,
                        stdev_W=conf.stdev_W, bias_routing=conf.bias_routing, sparse_threshold=conf.sparse_threshold,
                        sparsify=conf.sparsify, sparse_topk=conf.sparse_topk, hidden_capsules=conf.hidden_capsules)

    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, num_classes=label_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = CapsuleTrainer(model, capsule_loss, optimizer, data_train, data_test, conf)

    trainer.run()


if __name__ == "__main__":
    main()
