import torch
from torch.autograd import Variable
from torch import nn
from layers import Conv2dPrimaryLayer, DenseCapsuleLayer
from utils import one_hot, new_grid_size
import torch.nn.functional as F


class _CapsNet(nn.Module):
    """ Abstract CapsuleNet class."""

    def __init__(self, num_final_caps):
        super().__init__()
        self.num_final_caps = num_final_caps

    @staticmethod
    def compute_probs(caps):
        """ Compute class probabilities from the capsule/vector length.
        :param caps: capsules of shape [batch_size, num_capsules, dim_capsules]
        :returns probs of shape [batch_size, num_capsules]
        """
        return torch.sqrt((caps ** 2).sum(dim=-1, keepdim=False))

    @staticmethod
    def compute_predictions(probs):
        """Compute predictions by selecting
        :param probs: [batch_size, num_classes/capsules]
        :returns [batch_size]
        """
        _, index_max = probs.max(dim=1, keepdim=False)
        return index_max.squeeze()

    def create_decoder_input(self, final_caps, labels=None):
        """ Construct decoder input based on class probs and final capsules.
        Flattens capsules to [batch_size, num_final_caps * dim_final_caps] and sets all values which do not come from
        the correct class/capsule to zero (masks). During training the labels are used to masks, during inference the
        max of the class probabilities.
        :param labels: [batch_size, 1], if None: use predictions
        :return: [batch_size, num_final_caps * dim_final_caps]
        """
        targets = labels if type(labels) == Variable else self.compute_predictions(self.compute_probs(final_caps))
        masks = one_hot(targets, self.num_final_caps)
        ##todo check if cuda has to be called, make one hot for variables
        masked_caps = final_caps * masks[:, :, None]
        decoder_input = masked_caps.view(final_caps.shape[0], -1)
        return decoder_input

    def compute_acc(self, probs, label):
        """ Compute accuracy of batch
        :param probs: [batch_size, classes]
        :param label: [batch_size]
        :return: batch accurarcy (float)
        """
        return sum(self.compute_predictions(probs).data == label.data) / probs.size(0)


class BasicCapsNet(_CapsNet):
    """ Implements a CapsNet based using the architecture described in Dynamic Routing Hinton 2017.

    Input should be an image of shape: [batch_size, channels, width, height]

    Args:
        in_channels (int): the number of channels in the input
        digit_caps (int): the number of capsule in the final layer, the represent the output classes
        prim_caps (int): the number of capsules in the primary layer
        vec_len_prim (int): the dimensionality of the primary capsules
        vec_len_digit (int): the dimensionality of the digit capsules
        routing_iters (int): the number of iterations in the routing algorithm

    Returns:
        - class predictions of shape: [batch_size, num_classes/num_digit_caps]
        - image reconstruction of shape: [batch_size, channels, width, height]
    """

    def __init__(self, in_channels, digit_caps, vec_len_prim, vec_len_digit, routing_iters, prim_caps, in_height,
                 in_width):
        super().__init__(digit_caps)

        # initial convolution
        conv_channels = prim_caps * vec_len_prim
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=9, stride=1, padding=0,
                               bias=True)
        self.relu = nn.ReLU()

        # compute primary capsules
        self.primary_caps_layer = Conv2dPrimaryLayer(in_channels=conv_channels, out_channels=prim_caps, vec_len=vec_len_prim)

        # grid of multiple primary caps channels is flattend, number of new channels: grid * point * channels in grid
        new_height, new_width = new_grid_size(*new_grid_size(in_height, in_width, 1, 9, 0), 2, 9, 0)
        in_channels_dense_layer = new_height * new_width * prim_caps
        self.dense_caps_layer = DenseCapsuleLayer(in_channels_dense_layer, digit_caps, vec_len_prim,
                                                  vec_len_digit, routing_iters)

        self.decoder = CapsNetDecoder(vec_len_digit, digit_caps, in_channels, in_height, in_width)

    def forward(self, x, t=None):
        # apply conv layer
        conv1 = self.relu(self.conv1(x))

        # compute grid of capsules
        primary_caps = self.primary_caps_layer(conv1)

        # flatten to insert into dense layer
        b, c, w, h, m = primary_caps.shape
        primary_caps_flat = primary_caps.view(b, c*w*h, m)

        # compute digit capsules
        final_caps = self.dense_caps_layer(primary_caps_flat)

        probs = self.compute_probs(final_caps)

        decoder_input = self.create_decoder_input(final_caps, t)
        recon = self.decoder(decoder_input)

        return probs, recon, final_caps


class BaselineCNN(nn.Module):
    """
    Note: the paper (Hinton 2017) mentions a slightly different architecture than in the source code:
    https://github.com/Sarasra/models/blob/master/research/capsules/models/conv_model.py
    """
    def __init__(self, classes, in_channels, in_height, in_width):
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=5)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=5)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=5)
        self.fc1 = nn.Linear(128 * in_height, in_width, 328)
        self.fc2 = nn.Linear(328, 192)
        self.fc3 = nn.Linear(192, classes)

    def forward(self, x, training=False):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, kernel_size=2, stride=2, padding=0))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2, padding=0))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2, padding=0))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(F.dropout(self.fc3(x), training=training))
        return F.log_softmax(x, dim=1)


class CapsNetDecoder(nn.Module):

    def __init__(self, vec_len_digit, digit_caps, in_channels, in_height, in_width):
        super(CapsNetDecoder, self).__init__()

        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width

        self.flat_reconstruction = nn.Sequential(
            nn.Linear(vec_len_digit * digit_caps, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, in_channels * in_height * in_width),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flat_reconstruction(x).view(-1, self.in_channels, self.in_height, self.in_width)
        return x


if __name__ == '__main__':

    # create fake data and labels
    data = Variable(torch.randn(3, 1, 28, 28))
    y = Variable(torch.LongTensor([6, 3, 2])).view(3, 1)

    # apply module
    net = BasicCapsNet(digit_caps=10)
    class_probs, reconstructions = net(data, y)

    # checks
    assert(reconstructions.shape == torch.Size([3, 1, 28, 28]))
    assert(class_probs.shape == torch.Size([3, 10]))
    print("Checks passed")

