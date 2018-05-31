import torch
import torch.nn as nn
from utils import squash, init_weights, flex_profile, get_device


class DynamicRouting(nn.Module):

    def __init__(self, j, i, n, softmax_dim, bias_routing, sparse_threshold, sparsify):
        super().__init__()
        self.soft_max = torch.nn.Softmax(dim=softmax_dim)
        self.j = j
        self.i = i
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
        self.sparse_threshold = sparse_threshold
        self.sparsify = sparsify

    @flex_profile
    def forward(self, u_hat, iters):

        b = u_hat.shape[0]
        routing_stats = {"mask_rato": 0.0, "avg_neg_devs": 0.0, "max_neg_devs":0.0}

        if self.b_vec is None:
            self.b_vec = torch.zeros(b, self.j, self.i, device=get_device(),  requires_grad=False)
        b_vec = self.b_vec

        for index in range(iters):
            # softmax of i, weight of all predictions should sum to 1, note in tf code this does not give an error
            c_vec = self.soft_max(b_vec)

            # in einsum: bij, bjin-> bjn
            # in matmul: bj1i, bjin = bj (1i)(in) -> bjn
            s_vec = torch.matmul(c_vec.view(b, self.j, 1, self.i), u_hat).squeeze()
            if type(self.bias) == torch.nn.Parameter:
                s_vec_bias = s_vec + self.bias
            else:
                s_vec_bias = s_vec
            v_vec = squash(s_vec_bias)

            if index < (iters - 1):  # skip update last iter
                # in einsum: "bjin, bjn-> bij", inner product over n
                # in matmul: bji1n, bj1n1 = bji (1n)(n1) = bji1
                # note: use x=x+1 instead of x+=1 to ensure new object creation and avoid inplace operation
                b_vec = b_vec + torch.matmul(u_hat.view(b, self.j, self.i, 1, self.n),
                                             v_vec.view(b, self.j, 1, self.n, 1)).squeeze()

                # sparsify before last itter
                if index == (iters - 2) and self.sparsify:  #todo: make sure only to sparsify at last iteration
                    # todo include activaton
                    # activation = _CapsNet.compute_logits(v_vec)
                    #
                    avg_b_j = b_vec.sum(dim=2) / self.i

                    means = (avg_b_j.sum(dim=1) / self.j).view(-1, 1)
                    deviations = (avg_b_j - means)
                    neg_deviations = deviations * (deviations < 0).float()
                    avg_neg_deviations = neg_deviations.mean()
                    max_neg_deviations = neg_deviations.min(dim=1)[0].mean(dim=0)

                    a, _ = torch.max(avg_b_j, dim=1)
                    exponent = avg_b_j - a.view(-1, 1)
                    threshold = torch.tensor(self.sparse_threshold, device=get_device()).log() + a + torch.log(torch.exp(exponent).sum(dim=1))
                    delete_values = (avg_b_j < threshold.view(-1, 1))
                    mask_rato = len(delete_values[delete_values==0]) / (self.j * b)
                    # b_vec = keep_values.view(-1, self.j, 1) * b_vec
                    b_vec[delete_values, :] = float("-inf")

                    routing_stats["mask_rato"] = mask_rato
                    routing_stats["avg_neg_devs"] = avg_neg_deviations.item()
                    routing_stats["max_neg_devs"] = max_neg_deviations.item()
            else:
                if self.sparsify and iters > 1:
                    v_vec[delete_values, :] = 0
            if self.log_function:
                self.log_function(index, u_hat, b_vec, c_vec, v_vec, s_vec, s_vec_bias)
        return v_vec, routing_stats

    def _sparsify(self):
        """ Sparsifies b_vec"""


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

    def __init__(self, in_channels, out_channels, vec_len, squash_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vector_length = vec_len

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * vec_len, kernel_size=9, stride=2,
                              bias=True)
        self.conv = init_weights(conv)

        self.squash_dim = squash_dim

    @flex_profile
    def forward(self, input):
        """
        :param input: [b, c, h, w]
        :return: [b, c, h, w, vec]
        """
        features = self.conv(input)     # [b, out_c*vec_len, h, w)
        _, _, h, w = features.shape
        caps_raw = features.contiguous().view(-1, self.out_channels, self.vector_length, h, w)      # [b, c, vec, h, w]
        caps_raw = caps_raw.permute(0, 1, 3, 4, 2)  # [b, c, h, w, vec]

        return squash(caps_raw, dim=self.squash_dim)


class DenseCapsuleLayer(nn.Module):

    def __init__(self, in_capsules, out_capsules, vec_len_in, vec_len_out, routing_iters, stdev):
        super(DenseCapsuleLayer, self).__init__()

        self.in_capsules = in_capsules
        self.out_channels = out_capsules
        self.vector_len_in = vec_len_in
        self.vector_len_out = vec_len_out
        self.routing_iters = routing_iters

        self.W = nn.Parameter(stdev * torch.randn(1, out_capsules, in_capsules, vec_len_out, vec_len_in))

    @flex_profile
    def forward(self, input):
        batch_size = input.shape[0]
        input_ = input.view(batch_size, 1, self.in_capsules, self.vector_len_in, 1)
        u_hat = torch.matmul(self.W, input_).squeeze()
        return u_hat


class Conv2dCapsuleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Not implemented yet.")





