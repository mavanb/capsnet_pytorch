import pickle
from utils import get_device
from torch.nn.modules.loss import _Loss
import time
import numpy as np
import torch
from layers import DenseCapsuleLayer, DynamicRouting
from torch import nn
import os


class TestModel(nn.Module):

    def __init__(self, num_childs, num_parents, len_in, len_out, sparsify, sparse_topk):
        super().__init__()

        self.dense_layer = DenseCapsuleLayer(in_capsules=num_childs, out_capsules=num_parents,
                                        vec_len_in=len_in, vec_len_out=len_out,
                                        stdev=0.1)

        self.routing_layer = DynamicRouting(j=num_parents, i=num_childs, n=len_out, softmax_dim=1,
                                       bias_routing=True, sparse_threshold=0.99,
                                       sparsify=sparsify, sparse_topk=sparse_topk)

    def forward(self, data):
        """ Do forward pass and measure time.

        Args:
            data: (tensor) input data

        Returns:
            caps: (tensor) output capsules
            dense_time (float) time in seconds of the forward pass through DenseCapsuleLayer
            rout_time (float) time in seconds of the forward pass through DynamicRoutingLayer
        """
        start_dense = time.time()
        all_caps = self.dense_layer(data)
        dense_time = time.time() - start_dense

        start_rout = time.time()
        caps, _ = self.routing_layer(all_caps, 3)
        rout_time = time.time() - start_rout

        return caps, dense_time, rout_time


class FakeLoss(_Loss):
    def __init__(self, size_average=True):
        super(FakeLoss, self).__init__(size_average)

    def forward(self, predict, labels):
        return (predict - labels).sum()


class TimeClocker:
    """ TimeClocker is clocks the average time of a forward and backward pass through a specified model which
    consists of one DenseCapsule layer and one corresponding DynamicRouting layer.

    Experiments in speed_check.py showed that train time of Capsule Module (Dense + Rout) could be measured
    separately form the rest of the network. However seperating the Dense and Rout layer did not give consistent
    times for both the forward and the backward pass.

    Therefore, we time:
    - total time: dense and rout, forward and backward
    - dense_time: dense, forward
    - rout_time: rout, forward.
    """

    steps = 10
    max_childs = 200
    max_parents = 200
    len_in = 8
    len_out = 16
    sparsify = "None"
    sparse_topk = "0.1;0.3"

    batch_size = 128

    def __init__(self, sparse_topk_prev, out_file):
        """
        Args:
            sparse_topk_prev: (str) Sparsity applied to the layer before
            out_file: (str) Name of the output file
        """
        self.sparse_topk_prev = sparse_topk_prev

        # do not start at zero, as this causes problems
        self.child_iter = range(2, self.max_childs+3, self.steps)
        self.parent_iter = range(2, self.max_parents+3, self.steps)

        self.tot_parents = len(self.child_iter)
        self.tot_childs = len(self.parent_iter)

        # init array with time measurements
        self.time_array = np.zeros((self.tot_childs, self.tot_parents))

        self.out_file = out_file

    def clock(self):

        # loop over all childs
        for i, num_childs in enumerate(self.child_iter):

            # over all parents
            for j, num_parents in enumerate(self.parent_iter):

                # construct fake data, should have dimensions that fit number of childs and length child
                fake_data = torch.randn((self.batch_size, num_childs, self.len_in), requires_grad=False,
                                        dtype=torch.float32, device=get_device())

                # construct fake target, should have dimensions that fit number of parents and length parent
                fake_target = torch.zeros((self.batch_size, num_parents, self.len_out), requires_grad=False,
                                          dtype=torch.float32, device=get_device())

                # sparsify the the data with the given sparsity of the previous layer
                fake_data = self.sparsify_fake_date(fake_data, "nodes_topk", self.sparse_topk_prev)

                model = TestModel(num_childs, num_parents, self.len_in, self.len_out, self.sparsify, self.sparse_topk)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                loss = FakeLoss()

                model.to(get_device())
                torch.backends.cudnn.benchmark = True

                # compute the runtime of on e forward and backward pass
                total_time = self.eval_model(model, optimizer, loss, fake_data, fake_target)

                # col 0 is child column, col 1 is parent column
                self.time_array[i, j] = total_time

                print(f"\rParents: {j+1}/{self.tot_parents}, Childs: {i+1}/{self.tot_childs}", end="")

    @staticmethod
    def sparsify_fake_date(data, sparsify, sparse_topk):
        """ Run data through layer above which output shape is the same as input. This layer
        sparsifies the data.

        Args:
            data: (tensor) Non sparse data of shape bim
            sparsify: (str) Method to sparsify, should be None or nodes_topk
            sparse_topk: (str) topk parameter

        Returns: (tensor) Sparse data of shape bim
        """
        dense = DenseCapsuleLayer(in_capsules=data.shape[1], out_capsules=data.shape[1],
                                        vec_len_in=data.shape[2], vec_len_out=data.shape[2],
                                        stdev=0.1).to(get_device())

        rout = DynamicRouting(j=data.shape[1], i=data.shape[1], n=data.shape[2], softmax_dim=1,
                              bias_routing=True, sparse_threshold=0.99,
                              sparsify=sparsify, sparse_topk=sparse_topk).to(get_device())
        return rout(dense(data), 3)[0].data

    def save(self):
        if not os.path.exists("clock_times"):
            os.makedirs("clock_times")

        with open(f"clock_times/{self.out_file}.pkl", "wb") as outf:
            pickle.dump(self, outf)

    @staticmethod
    def eval_model(model, optimizer, fake_loss, fake_data, fake_labels):
        """ Measure the run time by running the model 2 times and take the average. The first iter seems is somehow
        always much slower, a sort of preheat phase. We skip this one.
        """

        time_list = []

        for _ in range(3):

            model.train()
            optimizer.zero_grad()

            total_start = time.time()

            predict, dense_time, rout_time = model(fake_data)
            loss = fake_loss(predict, fake_labels)

            loss.backward()
            optimizer.step()

            total_time = (time.time() - total_start) / fake_data.shape[0]

            time_list.append(total_time)

        # the first one is somehow much quicker, drop this one
        time_list = time_list[1:]

        total_avg = np.asarray(time_list).mean() * 1000
        return total_avg


if __name__ == "__main__":

    t = TimeClocker("0.0;0.5", "output_50")
    t.clock()
    t.save()

    t = TimeClocker("0.0;0.0", "output_0")
    t.clock()
    t.save()



