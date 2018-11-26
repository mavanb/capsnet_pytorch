"""

This module checks whether the run times of modules in capsule network can be measured separately.

To do so, we train:
- a three layer capsule layer
- the same capsule network with the second layer removed
- the second layer separately
- additionally, we check the two components of the second layer: dense layer, routing layer

"""

from __future__ import print_function

import torch
from torch import nn
from torchvision import transforms

from configurations.conf import get_conf, capsule_arguments
from data.data_loader import get_dataset
from capsule_trainer import CapsuleTrainer
from layers import Conv2dPrimaryLayer, DenseCapsuleLayer, DynamicRouting
from loss import CapsuleLoss, _Loss
from nets import _CapsNet, CapsNetDecoder
from utils import get_logger
from utils import new_grid_size, init_weights


class TripleOrDoubleCapsNet(_CapsNet):

    def __init__(self, in_channels, digit_caps, vec_len_prim, vec_len_digit, routing_iters, prim_caps, in_height,
                 in_width, softmax_dim, squash_dim, stdev_W, bias_routing, sparse_threshold, sparsify, sparse_topk,
                 hidden_capsules, extra_layer=False):
        super().__init__(digit_caps)

        self.extra_layer = extra_layer
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
                                               sparse_method=sparsify, mask_rato=sparse_topk)

        self.dense_caps_layer2 = DenseCapsuleLayer(i=hidden_capsules, j=hidden_capsules,
                                                   m=vec_len_digit, n=vec_len_digit,
                                                   stdev=stdev_W)

        self.dynamic_routing2 = DynamicRouting(j=hidden_capsules, i=hidden_capsules, n=vec_len_digit, softmax_dim=softmax_dim,
                                               bias_routing=bias_routing, sparse_threshold=sparse_threshold,
                                               sparse_method=sparsify, mask_rato=sparse_topk)

        self.dense_caps_layer3 = DenseCapsuleLayer(i=hidden_capsules, j=digit_caps,
                                                   m=vec_len_digit, n=vec_len_digit,
                                                   stdev=stdev_W)

        self.dynamic_routing3 = DynamicRouting(j=digit_caps, i=hidden_capsules, n=vec_len_digit, softmax_dim=softmax_dim,
                                               bias_routing=bias_routing, sparse_threshold=sparse_threshold,
                                               sparse_method=sparsify, mask_rato=sparse_topk)

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

        all_caps = self.dense_caps_layer1(primary_caps_flat)
        caps, _ = self.dynamic_routing1(all_caps, self.routing_iters)

        if self.extra_layer:
            all_caps = self.dense_caps_layer2(caps)
            caps, _ = self.dynamic_routing2(all_caps, self.routing_iters)

        all_caps = self.dense_caps_layer3(caps)
        final_caps, stats = self.dynamic_routing3(all_caps, self.routing_iters)

        logits = self.compute_logits(final_caps)

        decoder_input = self.create_decoder_input(final_caps, t)
        recon = self.decoder(decoder_input)

        return logits, recon, final_caps, stats


class FakeLoss(_Loss):
    def __init__(self, size_average=True):
        super(FakeLoss, self).__init__(size_average)

    def forward(self, images, labels):
        return (images - labels).sum()


def test_route_time(model, conf):
    import copy
    from utils import get_device
    from torch.nn.modules.loss import _Loss
    import time
    import numpy as np

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # create fake data
    fake_data_dense = torch.randn((conf.batch_size, model.dense_caps_layer2.i, model.dense_caps_layer2.m), requires_grad=False, dtype=torch.float32, device=get_device())
    fake_data_rout = torch.zeros((conf.batch_size, model.dense_caps_layer2.j, model.dense_caps_layer2.i, model.dense_caps_layer2.n), requires_grad=False, dtype=torch.float32, device=get_device())
    fake_target_dense = torch.zeros_like(fake_data_rout)
    fake_target_rout = torch.zeros_like(fake_data_dense)

    fake_loss = FakeLoss()

    # init time lists
    total_time_list = []
    dense_time_list = []
    rout_time_list = []

    # loop over all tests
    for time_list, mode in [(total_time_list, "total"), (dense_time_list, "dense"), (rout_time_list, "rout")]:

        # copy layer 2 of orginal tripleordouble capsnet
        dense_layer2 = copy.deepcopy(model.dense_caps_layer2)
        routing_layer2 = copy.deepcopy(model.dynamic_routing2)

        for _ in range(10):

            optimizer.zero_grad()

            start = time.time()

            if mode == "total":
                all_caps = dense_layer2(fake_data_dense)
                caps, _ = routing_layer2(all_caps, 3)
                loss = fake_loss(caps, fake_target_rout)
            elif mode == "dense":
                all_caps = dense_layer2(fake_data_dense)
                loss = fake_loss(all_caps, fake_target_dense)
            elif mode == "rout":
                caps, _ = routing_layer2(fake_data_rout, 3)
                loss = fake_loss(caps, fake_target_rout)
            else:
                print("Mode does not exists")

            loss.backward()
            optimizer.step()

            diff_time = (time.time() - start) / fake_data_dense.shape[0]

            time_list.append(diff_time)

    total_avg = np.asarray(total_time_list).mean() * 1000
    rout_avg = np.asarray(rout_time_list).mean() * 1000
    dense_avg = np.asarray(dense_time_list).mean() * 1000

    return total_avg, dense_avg, rout_avg


def main():
    custom_args = capsule_arguments("capsnet", path_root="..")
    conf, parser = get_conf([custom_args], path_root="..")

    log = get_logger(__name__)
    log.info(parser.format_values())
    transform = transforms.ToTensor()
    data_train, data_test, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)

    conf.epochs = 1
    log.info("Set epochs to 1. Speed check runs for only for 1 epoch.")

    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, include_recon=conf.use_recon)

    print("\n --- Measure training time of Double CapsNet --- \n")
    model = TripleOrDoubleCapsNet(in_channels=data_shape[0], digit_caps=label_shape, vec_len_prim=8, vec_len_digit=16,
                                  routing_iters=conf.routing_iters, prim_caps=conf.prim_caps, in_height=data_shape[1],
                                  in_width=data_shape[2], softmax_dim=conf.softmax_dim, squash_dim=conf.squash_dim,
                                  stdev_W=conf.stdev_W, bias_routing=conf.bias_routing,
                                  sparse_threshold=conf.sparse_threshold,
                                  sparsify=conf.sparsify, sparse_topk=conf.sparse_topk,
                                  hidden_capsules=conf.hidden_capsules, extra_layer=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = CapsuleTrainer(model, capsule_loss, optimizer, data_train, data_test, conf)
    trainer.run()

    print("\n --- Measure training time of the extra layer --- \n")
    total_avg, dense_avg, rout_avg = test_route_time(model, conf)
    log.info(f"Time layer 2: {total_avg:0.6f}")
    log.info(f"Time dense layer 2: {dense_avg:0.6f}")
    log.info(f"Time rout layer 2: {rout_avg:0.6f}")

    print("\n --- Measure training time of Triple CapsNet --- \n")
    model = TripleOrDoubleCapsNet(in_channels=data_shape[0], digit_caps=label_shape, vec_len_prim=8, vec_len_digit=16,
                                  routing_iters=conf.routing_iters, prim_caps=conf.prim_caps, in_height=data_shape[1],
                                  in_width=data_shape[2], softmax_dim=conf.softmax_dim, squash_dim=conf.squash_dim,
                                  stdev_W=conf.stdev_W, bias_routing=conf.bias_routing,
                                  sparse_threshold=conf.sparse_threshold,
                                  sparsify=conf.sparsify, sparse_topk=conf.sparse_topk,
                                  hidden_capsules=conf.hidden_capsules, extra_layer=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = CapsuleTrainer(model, capsule_loss, optimizer, data_train, data_test, conf)
    trainer.run()




if __name__ == "__main__":
    main()
