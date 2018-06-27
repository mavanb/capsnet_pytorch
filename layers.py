import torch
import torch.nn as nn
from utils import squash, init_weights, flex_profile, get_device, calc_entropy, batched_index_select, multinomial_nd
import math

class DynamicRouting(nn.Module):

    def __init__(self, j, n, bias_routing, sparse_method, sparse_target, mask_percent):
        super().__init__()
        self.soft_max = torch.nn.Softmax(dim=1)
        self.j = j
        self.n = n

        # init depends on batch_size which depends on input size, declare dynamically in forward. see:
        # https://discuss.pytorch.org/t/dynamic-parameter-declaration-in-forward-function/427/2
        self.b_vec = None

        if bias_routing:
            b_routing = nn.Parameter(torch.zeros(j, n))
            b_routing.data.fill_(0.1)
            self.bias = b_routing
        else:
            self.bias = None

        # that can be implemented to enable analysis at end of each routing iter
        self.log_function = None
        self.mask_percent = mask_percent
        self.sparse_method = sparse_method
        self.sparse_target = sparse_target

    @flex_profile
    def forward(self, u_hat, iters):

        # check if enough topk values ratios are given in the configuration
        if self.sparse_method != "None":
            assert len(self.mask_percent) + 1 >= iters, "Please specify for each update routing iter the sparse top k."\
                " Example: routing iters: 3, sparse_topk = 0.4-0.4"

        b = u_hat.shape[0]
        i = u_hat.shape[2]

        # todo: previously b_vec as only init once, but than the class must always have the same input shape
        # todo: it seems that init does not cost much time, so temporily do it this way
        # if self.b_vec is None:
            # self.b_vec = torch.zeros(b, self.j, self.i, device=get_device(),  requires_grad=False)
        self.b_vec = torch.zeros(b, self.j, i, device=get_device(), requires_grad=False)
        b_vec = self.b_vec

        # track entropy of c_vec per iter
        entropy_layer = torch.zeros(b, iters, device=get_device(), requires_grad=False)

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

            v_vec = squash(s_vec_bias)

            if index < (iters - 1):  # skip update last iter
                # in einsum: "bjin, bjn-> bij", inner product over n
                # in matmul: bji1n, bj1n1 = bji (1n)(n1) = bji1
                # note: use x=x+1 instead of x+=1 to ensure new object creation and avoid inplace operation
                b_vec = b_vec + torch.matmul(u_hat.view(b, self.j, i, 1, self.n),
                                             v_vec.view(b, self.j, 1, self.n, 1)).view(b, self.j, i)

                if self.sparse_method != "none":
                    b_vec = self.sparsify(b_vec, index, self.sparse_target, self.sparse_method, i)

            if self.log_function:
                self.log_function(index, u_hat, b_vec, c_vec, v_vec, s_vec, s_vec_bias)
        return v_vec, entropy_layer

    def sparsify(self, b_vec, iter, target, method, i):

        # get batch_size
        b = b_vec.shape[0]

        # number of elements to select (or not mask)
        select_count = self.j
        for prev in self.mask_percent[0:iter+1]:
            select_count = math.ceil(select_count * (1 - prev))

        # select_count == j means select all, thus skip
        if select_count < self.j:

            # construct mask the expected c_vec if not sparsified
            c_vec_temp = self.soft_max(b_vec)

            if target == "nodes":
                c_vec_temp = c_vec_temp.sum(dim=2) / i

            # create empty selected tensor
            selected = torch.zeros_like(c_vec_temp, requires_grad=False, device=get_device(), dtype=torch.uint8)

            if method == "topk":

                # take topk largest values
                _, indices = torch.topk(c_vec_temp, select_count, largest=True, sorted=False, dim=1)

                selected.scatter_(1, indices, 1)

            else:

                if method == "sample":
                    weights = c_vec_temp

                elif method == "random":

                    # set uniform weights
                    weights = torch.ones_like(c_vec_temp, requires_grad=False, device=get_device())

                    # set weights of previous selected to zero
                    if iter > 0:
                        prev_inf_mask = c_vec_temp == 0.0
                        weights[prev_inf_mask] = 0.0
                else:
                    raise ValueError("Selected method does not exists.")

                indices = multinomial_nd(weights, select_count, dim=1, replacement=False)

                selected.scatter_(1, indices, 1)

            if target == "nodes":
                selected = selected.view(b, self.j, 1).expand(b, self.j, i)

            mask = selected ^ 1

            b_vec[mask] = float("-inf")

        return b_vec


class LinearPrimaryLayer(nn.Module):

    def __init__(self, in_features, out_capsules, vec_len):
        super().__init__()
        self.out_capsules = out_capsules
        self.vec_len = vec_len
        self.linear = nn.Linear(in_features, out_capsules * vec_len, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return squash(x.view(-1, self.out_capsules, self.vec_len))


class Conv2dPrimaryLayer(nn.Module):

    def __init__(self, in_channels, out_channels, vec_len):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vector_length = vec_len

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * vec_len, kernel_size=9, stride=2,
                              bias=True)
        self.conv = init_weights(conv)


    def forward(self, input):
        """
        :param input: [b, c, h, w]
        :return: [b, c, h, w, vec]
        """
        features = self.conv(input)     # [b, out_c*vec_len, h, w)
        _, _, h, w = features.shape
        caps_raw = features.contiguous().view(-1, self.out_channels, self.vector_length, h, w)      # [b, c, vec, h, w]
        caps_raw = caps_raw.permute(0, 1, 3, 4, 2)  # [b, c, h, w, vec]

        # squash on the vector dimension
        return squash(caps_raw, dim=2)


class DenseCapsuleLayer(nn.Module):

    def __init__(self, i, j, m, n, stdev):
        super(DenseCapsuleLayer, self).__init__()

        self.i = i
        self.j = j
        self.m = m
        self.n = n

        self.W = nn.Parameter(stdev * torch.randn(1, j, i, n, m))

    def forward(self, input):
        b,i,m = input.shape
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
            #todo change print to log
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


class Conv2dCapsuleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Not implemented yet.")





