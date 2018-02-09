import torch
import torch.nn as nn
from utils import dynamic_routing, squash, parameter


class Conv2dPrimaryLayer(nn.Module):

    def __init__(self, in_channels, out_channels, vec_len):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vector_length = vec_len

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * vec_len, kernel_size=9, stride=2,
                              bias=True)

    def forward(self, input):
        """
        :param input: [b, c, h, w]
        :return: [b, c, h, w, vec]
        """

        features = self.conv(input)     # [b, out_c*vec_len, h, w)
        _, _, h, w = features.shape
        caps_raw = features.contiguous().view(-1, self.out_channels, self.vector_length, h, w)      # [b, c, vec, h, w]
        caps_raw = caps_raw.permute(0, 1, 3, 4, 2)  # [b, c, h, w, vec]
        return squash(caps_raw)


class DenseCapsuleLayer(nn.Module):

    def __init__(self, in_channels, out_channels, vec_len_in, vec_len_out, routing_iters):
        super(DenseCapsuleLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vector_len_in = vec_len_in
        self.vector_len_out = vec_len_out
        self.routing_iters = routing_iters

        self.W = parameter(torch.randn(1, out_channels, in_channels, vec_len_out, vec_len_in))

    def forward(self, input):
        batch_size = input.shape[0]
        input_ = input.view(batch_size, 1, self.in_channels, self.vector_len_in, 1)
        u_hat = torch.matmul(self.W, input_).squeeze()
        return dynamic_routing(u_hat, self.routing_iters)


class Conv2dCapsuleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Not implemented yet.")


class DensePrimaryLayer(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Not implemented yet.")

