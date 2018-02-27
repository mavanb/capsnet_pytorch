import torch
import torch.nn as nn
from utils import variable, squash, parameter


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

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * vec_len, kernel_size=9, stride=2,
                              bias=True)

        self.squash_dim = squash_dim

    def forward(self, input):
        """
        :param input: [b, c, h, w]
        :return: [b, c, h, w, vec]
        """
        features = self.conv(input)     # [b, out_c*vec_len, h, w)
        _, _, h, w = features.shape
        caps_raw = features.contiguous().view(-1, self.out_channels, self.vector_length, h, w)      # [b, c, vec, h, w]
        caps_raw = caps_raw.permute(0, 1, 3, 4, 2)  # [b, c, h, w, vec]

        # squash dim is supposed to be 2, but 1 seems too work way way beter
        return squash(caps_raw, dim=self.squash_dim)


class DenseCapsuleLayer(nn.Module):

    def __init__(self, in_capsules, out_capsules, vec_len_in, vec_len_out, routing_iters):
        super(DenseCapsuleLayer, self).__init__()

        self.in_capsules = in_capsules
        self.out_channels = out_capsules
        self.vector_len_in = vec_len_in
        self.vector_len_out = vec_len_out
        self.routing_iters = routing_iters

        # self.W = parameter(torch.randn(1, out_capsules, in_capsules, vec_len_out, vec_len_in))
        self.W = parameter(torch.randn(1, in_capsules, out_capsules, vec_len_out, vec_len_in))
        # todo change back: changed for check (permutation)

        ## todo remove this
        self.W.data.fill_(0.01)

    def forward(self, input):
        batch_size = input.shape[0]
        input_ = input.view(batch_size, 1, self.in_capsules, self.vector_len_in, 1)
        input_ = input_.permute(0, 2, 1, 3, 4) #todo change back (permutation)
        u_hat = torch.matmul(self.W, input_).squeeze()
        u_hat = u_hat.permute(0, 2, 1, 3)
        return u_hat


class Conv2dCapsuleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Not implemented yet.")





