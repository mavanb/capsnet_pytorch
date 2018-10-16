""" Layer module.

This module contains all layers used in the network module.

References:
    [1] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing between capsules,” in NIPS, pp. 3859–3869, 2017.

"""

import torch
import torch.nn as nn
from utils import squash, init_weights, flex_profile, get_device, calc_entropy, batched_index_select, multinomial_nd
import math


class DynamicRouting(nn.Module):
    """ Dynamic routing procedure.

    Routing-by-agreement as in [1], extended by several optional sparse softmax functions.

    Args:
        j (int): Number of parent capsules.
        n (int): Vector length of the parent capsules.
        bias_routing (bool): Add a bias parameter to the average parent predictions.
        sparse (SparseMethods): Sparsity methods to be used.
    """

    def __init__(self, j, n, bias_routing, sparse):  # sparse_method, sparse_target, mask_percent):
        super().__init__()
        self.soft_max = torch.nn.Softmax(dim=1)
        self.j = j
        self.n = n

        self.sparse = sparse

        # init depends on batch_size which depends on input size, declare dynamically in forward. see:
        # https://discuss.pytorch.org/t/dynamic-parameter-declaration-in-forward-function/427/2
        self.b_vec = None

        # init bias parameter
        if bias_routing:
            b_routing = nn.Parameter(torch.zeros(j, n))
            b_routing.data.fill_(0.1)
            self.bias = b_routing
        else:
            self.bias = None

        # log function that is called in the forward pass to enable analysis at end of each routing iter
        self.log_function = None

    @flex_profile
    def forward(self, u_hat, iters):
        """ Forward pass

        Args:
            u_hat (FloatTensor): Prediction vectors of the child capsules for the parent capsules. Shape: [batch_size,
                num parent caps, num child caps, len final caps]
            iters (int): Number of routing iterations.

        Returns:
            v_vec (FloatTensor): Tensor containing the squashed average predictions using the routing weights of the
                routing weight update. Shape: [batch_size, num parent capsules, len parent capsules]
            entropy_layer: Average routing entropy of all capsules in the layer per routing iteration. Shape:
                [batch_size, routing iters].
        """

        b = u_hat.shape[0]  # batch_size
        i = u_hat.shape[2]  # number of parent capsules

        # init empty b_vec, on init would be better, but b and i are unknown there. Takes hardly any time this way.
        self.b_vec = torch.zeros(b, self.j, i, device=get_device(), requires_grad=False)
        b_vec = self.b_vec

        # track entropy of c_vec per iter
        entropy_layer = torch.zeros(b, iters, device=get_device(), requires_grad=False)

        # loop over all routing iterations
        for index in range(iters):

            # softmax over j, weight of all predictions should sum to 1
            c_vec = self.soft_max(b_vec)

            # compute entropy of weight distribution of all capsules
            # stats.append(calc_entropy(c_vec, dim=1).mean().item())
            entropy_layer[:, index] = calc_entropy(c_vec, dim=1).mean(dim=1)

            # created unsquashed prediction for parents capsules by a weighted sum over the child predictions
            # in einsum: bij, bjin-> bjn
            # in matmul: bj1i, bjin = bj (1i)(in) -> bjn
            s_vec = torch.matmul(c_vec.view(b, self.j, 1, i), u_hat).squeeze()

            # add bias to s_vec
            if type(self.bias) == torch.nn.Parameter:
                s_vec_bias = s_vec + self.bias

                # don't add a bias to capsules that have no activation add all
                # check which capsules where zero
                reset_mask = (s_vec.sum(dim=2) == 0)

                # set them back to zero again
                s_vec_bias[reset_mask, :] = 0
            else:
                s_vec_bias = s_vec

            # squash the average predictions
            v_vec = squash(s_vec_bias)

            # skip update last iter
            if index < (iters - 1):

                # compute the routing logit update
                # in einsum: "bjin, bjn-> bij", inner product over n
                # in matmul: bji1n, bj1n1 = bji (1n)(n1) = bji1
                b_vec_update = torch.matmul(u_hat.view(b, self.j, i, 1, self.n),
                                             v_vec.view(b, self.j, 1, self.n, 1)).view(b, self.j, i)

                # update b_vec
                # use x=x+1 instead of x+=1 to ensure new object creation and avoid inplace operation
                b_vec = b_vec + b_vec_update

                # loop over the sparse methods
                for step in self.sparse:
                    if step["target"] != "none":
                        try:
                            percent = step["percent"][index]
                        except IndexError:
                            raise IndexError(f"Expected {iters} values, only {len(step['percent'])} where given.")
                        b_vec = self.sparsify(b_vec, index, step["target"], step["method"], percent, i)

            # call log function every routing iter for optional analysis
            if self.log_function:
                self.log_function(index, u_hat, b_vec, c_vec, v_vec, s_vec, s_vec_bias)

        return v_vec, entropy_layer

    def sparsify(self, b_vec, iter, target, method, percent, i):

        # get batch_size
        b = b_vec.shape[0]

        # amount of edges left of the childs, min and max can be different due to nodes and edges sparse
        inf_count = (b_vec != float("-inf")).sum(dim=1)
        min_left = inf_count.min().item()
        max_left = inf_count.max().item()

        #  number of edges to select (thus not mask)
        select_count = math.ceil(min_left * (1 - percent))

        # select_count == j means select all, thus skip
        if select_count < max_left:

            # construct mask the expected c_vec if not sparsified
            # the sample method requires these
            c_vec_temp = self.soft_max(b_vec)

            # if node sparse, sum over the child capsules to get incoming weight of each parent
            if target == "nodes":
                c_vec_temp = c_vec_temp.sum(dim=2) / i

            # create empty selected tensor, byte tensor that indicates if weights are selected
            selected = torch.zeros_like(c_vec_temp, requires_grad=False, device=get_device(), dtype=torch.uint8)

            if method == "topk":

                # take the indices of the topk largest values
                _, indices = torch.topk(c_vec_temp, select_count, largest=True, sorted=False, dim=1)

                # write these indices to the selected tensor
                selected.scatter_(1, indices, 1)

            else:

                # when sampling, use the non sparse distribution as weights
                if method == "sample":
                    weights = c_vec_temp

                elif method == "random":

                    # set uniform weights
                    weights = torch.ones_like(c_vec_temp, requires_grad=False, device=get_device())

                    # set weights of previous selected to zero
                    if min_left < self.j:
                        prev_inf_mask = c_vec_temp == 0.0
                        weights[prev_inf_mask] = 0.0
                else:
                    raise ValueError(f"Select method '{method}' does not exists.")

                # sample the indices using the weights without replacement (random subset)
                indices = multinomial_nd(weights, select_count, dim=1, replacement=False)

                # write these indices to the selected tensor
                selected.scatter_(1, indices, 1)

            # if sparse nodes, expand back to the original weight tensor shape
            if target == "nodes":
                selected = selected.view(b, self.j, 1).expand(b, self.j, i)

            # mask all not selected weights
            mask = selected ^ 1

            # set these to -inf, is the same setting them to 0 and leaving them from the softmax
            b_vec[mask] = float("-inf")

        return b_vec


class LinearPrimaryLayer(nn.Module):
    """ Create primary capsules with one linear layer.

    Args:
        in_features (int): number of features in the input
        out_capsules (int): number of parent capsules
        vec_len (int): vector length of the parent capsules
    """

    def __init__(self, in_features, out_capsules, vec_len):
        super().__init__()
        self.out_capsules = out_capsules
        self.vec_len = vec_len
        self.linear = nn.Linear(in_features, out_capsules * vec_len, bias=True)

    def forward(self, x):
        """Forward pass. """
        x = self.linear(x)
        return squash(x.view(-1, self.out_capsules, self.vec_len))


class Conv2dPrimaryLayer(nn.Module):
    """ Compute grid of capsule by convolution layers.

    Create primary capsules as in [1]. The primary capsules are computed by:
     - first applying a conv layer with ReLU non-linearity to the input image
     - then applying a conv layer without non-linearity, reshape to capsules and apply squah non-linearity

    Args:
        in_channels (int): Number of channels of the input data/image.
        out_channels (int): Number of the capsules in the output grid.
        vec_len (int): Vector length of the primary capsules.
    """

    def __init__(self, in_channels, out_channels, vec_len):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vector_length = vec_len

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * vec_len, kernel_size=9, stride=2,
                         bias=True)
        self.conv = init_weights(conv)

    def forward(self, input):
        """ Forward pass

        Args:
            input (FloatTensor): Input image of shape [batch_size, in_channels, height_input, width_input]

        Returns:
            caps_raw (FloatTensor): Primary capsules in grid of shape
                [batch_size, out_channels, width grid, height grid, vec_len].
        """
        features = self.conv(input)
        _, _, h, w = features.shape
        caps_raw = features.contiguous().view(-1, self.out_channels, self.vector_length, h, w)  # [b, c, vec, h, w]
        caps_raw = caps_raw.permute(0, 1, 3, 4, 2)  # [b, c, h, w, vec]

        # squash on the vector dimension
        return squash(caps_raw)


class DenseCapsuleLayer(nn.Module):
    """ Dense Capsule Layer

    Dense capsule layer as in [1], but with optimized computation of the predictions if some of the child
    capsule are completely non-active.

    Args:
        i (int): Number of child capsules.
        j (int): Number of parent capsules.
        m (int): Vector length of the child capsules.
        n (int): Vector length of the parent capsules.
        stdev (float): Weight initialization transformation matrices.
    """

    def __init__(self, i, j, m, n, stdev):
        super(DenseCapsuleLayer, self).__init__()

        self.i = i
        self.j = j
        self.m = m
        self.n = n

        self.W = nn.Parameter(stdev * torch.randn(1, j, i, n, m))

    def forward(self, input):
        """ Forward pass

        Args:
            input (FloatTensor): Child capsules of the layer. Shape: [batch_size, i, n].

        Returns:
            FloatTensor: Tensor with predictions for each parent capsule of each non-zero child capsules. Shape:
                [batch_size, j, num non-zero child capsules, m].


        """
        b, i, m = input.shape
        n = self.n
        j = self.j
        assert i == self.i, "Unexpected number of childs as input"
        assert m == self.m, "Unexpected vector lenght as input"

        # the zero rows (m index all zero) in the input
        zero_rows = (input.sum(dim=2) == 0.0).sum(dim=1)

        # check number of zeros in first row
        zero_count = zero_rows[0].item()

        # check if number of zeros is consistens over batch
        valid = ((zero_rows != zero_count).sum() == 0).item()

        if zero_count > 0 and valid:

            non_zero_count = i - zero_count

            # chech which rows are not zero, and put in format to use for batch_index_select
            select_idx = (input.sum(dim=2) != 0.0).nonzero()[:, 1].view(b, non_zero_count)

            # expand W such that we can multiply batch elements with its own truncated W
            W = self.W.expand(b, self.j, self.i, self.n, self.m)
            W = batched_index_select(W, 2, select_idx)
            input = batched_index_select(input, 1, select_idx)

            # set new i to non zero count
            i_new = non_zero_count

        elif zero_count > 0:
            # in a very rare case it not valid (consistent over batch, apparently some row got full zero
            # by pure coincidence. log it and continue.
            print("Invalid sparse speed up. Did not apply speed up in this batch.")
            W = self.W
            i_new = i

        else:
            W = self.W
            i_new = i

        input = input.view(b, 1, input.shape[1], self.m, 1)

        # W: bjinm or 1jinm
        # input: b1jm1
        # matmul: bji(nm) * b1j(m1) = bjin1
        u_hat = torch.matmul(W, input).view(b, j, i_new, n)

        return u_hat
