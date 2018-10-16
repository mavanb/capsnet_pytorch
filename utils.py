""" Utils

General functions used throughout the project.

References:
    [1] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing between capsules,” in NIPS, pp. 3859–3869, 2017.
"""

import torch
import math
import logging
import sys


def flex_profile(func):
    """ Decorator for the @proflile decorator of kernprof.

    Slightly hacky solution to have avoid having to remove the profile decorator of kernprof all the time. To profile
    run: 'kernprof -v -l train_capsnet.py'. Make sure to first put @profile (should be removed afterwards) or
    @flex_profile (no need to remove) decorators above the functions you want to profile. Decorator does hardly hardly
    influences computation time, but use for nested functions that are called very often.
    """
    try:
        func = profile(func)
    except NameError:
        pass
    return func


def new_grid_size(grid, kernel_size, stride=1, padding=0):
    """ Calculate new images size after convoling.

    Function calculated the size of the grid after convoling an image or feature map. Used formula from:
    https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/

    Args:
        grid (tuple of ints): Tuple with 2 ints of the dimensions of the orginal grid size.
        kernel_size (int): Size of the kernel (is a square).
        stride (int, optional): Stride used.
        padding (int, optional): Padding used.
    """
    def calc(x): return int((x - kernel_size + 2 * padding)/stride + 1)
    return calc(grid[0]), calc(grid[1])


def padding_same_tf(grid, kernel, stride):
    """ Calculate the padding to mimic tensorflow

    TensorFlow padding SAME corresponds to padding such that: output size = ceil(input/stride)

    Args:
        grid: (tuple of ints): Tuple with 2 ints of the dimensions of the orginal grid size.
        kernel: (int): Size of the kernel (is a square).
        stride: (int, optional): Stride used.

    Returns
        int: padding used if the padding SAME of tensorflow was used.
    """
    def calc(x):
        out_size = math.ceil(x / stride)
        return int(((out_size - 1) * stride + kernel - x) / 2)
    return calc(grid[0]), calc(grid[1])


def squash(tensor, dim=-1):
    """ Squash function as defined in [1].

    Args:
        tensor (Tensor): Input tensor.
        dim (int, optional): Dimension on which to apply the squash function. Vector dimension. Defaults to the last.
    """
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1. + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-7)


def calc_entropy(input_tensor, dim):
    """ Calculate entropy over dimension of tensor. The sum over this dimension should thus always sum to one.

    Args:
        input_tensor: (tensor) Input tensor.
        dim (int): Dimension over which the entropy is calculated. All entries over this dimension should sum to 1.

    Returns:
    """
    # check if dimension makes sense
    assert dim < len(input_tensor.shape), "Dimension {} does not exists".format(dim)

    # slice should represent a probablity distribution, thus sum to one
    assert (get_approx_value(input_tensor.sum(dim=dim), 1, 1e2) == False).nonzero().shape == torch.Size([0]), "Dimension {} " \
        "does not sum to 1 everywhere".format(dim)

    # return entropy: H(p(x)) = sum_i p(x=i) log(1/p(x=i))
    return (input_tensor * (1/(input_tensor+1e-8)).log2()).sum(dim=dim)


def get_approx_value(input_tensor, value, precision=1e6):
    """ Get all entries that equal a certain value approximately.

    Args:
        input_tensor (tensor): Tensor to compare against of any size.
        value (float): Value to compare to.
        precision (float): Precision of comparision.

    Returns:
        ByteTensor: bool tensor of same size as input indication which entries are approx equal
    """
    return (input_tensor * precision).round() == value * precision


def one_hot(labels, depth):
    """ Create one-hot encoding matrix from vector of labels/indices.

    PyTorch does not have a one-hot function like tensorflow.

    Args:
        labels (LongTensor): Tensor of labels of shape: [batch_size].
        depth (int): Output length of one hot vectors i.e. number of classes.

    Returns:
        FloatTensor: Tensor of shape [batch_size, depth] with a one-hot representation of the labels.
    # """
    return torch.eye(depth, device=get_device()).index_select(dim=0, index=labels)


def init_weights(module, weight_mean=0, weight_stddev=0.1, bias_mean=0.1):
    """ Init weights of torch.module. """
    module.weight.data.normal_(weight_mean, weight_stddev)
    module.bias.data.fill_(bias_mean)
    return module


def get_device():
    """ Get the device on which running."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name):
    """Get info logger that logs to stdout."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def batched_index_select(input, dim, index):
    """ Select indices batch_wise.

    Code based on:
        https://discuss.pytorch.org/t/batched-index-select/9115/8
        https://stackoverflow.com/questions/49104307/indexing-on-axis-by-list-in-pytorch

    Args:
        input: (tensor) input tensor of size: batch x .. x N x ..
        dim: (int) dimension of the indices
        index: (tensor) the tensor with the indices of size: batch x M

    Returns: (tensor) selected tensor of size: batch x .. x M x ..

    """
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def multinomial_nd(input, num_samples, dim, replacement=False):
    """ Sample from a multinomial distribution.

    PyTorch does not support sampling form a multinomial distribution of more than 2 dimensions and has no dim argument
    (in version v0.4). Therefore, for 2 dimensions, we use a tensor permutation. For 3 dimensions we use a permutation
    and a for loop. Remains a relatively light computation.

    Args:
        input (Tensor): Input tensor.
        num_samples: Number of samples to sample.
        dim (int): Dimension which contains the parameters of the multinomial distribution.
        replacement (bool, optional): Sample with replacement yes/no.

    Returns:
        LongTensor: Tensor with the indices of the sampled categories and how often.

    """
    assert isinstance(dim, int), "dim should be int"
    assert num_samples <= input.shape[dim], "Num samples should be smaller than length of input in the sample dimension"

    d = len(input.shape)
    assert 1 <= d <= 3, "multinomial_nd only supports 1 to 3 dims"

    if d == 1:
        assert dim == 0, "dim should be 0"

        input = torch.multinomial(input, num_samples, replacement=replacement)

    elif d == 2:

        assert 0 <= dim <= 1, "dim should be 0 or 1"

        if dim == 0:
            input = input.permute(1, 0)

        input = torch.multinomial(input, num_samples, replacement=replacement)

        if dim == 0:
            input.permute(1, 0)

    elif d == 3:

        assert 0 <= dim <= 2, "dim should be 0, 1 or 2"

        # make sure indexed dimension is the last dimension
        if dim == 1:
            input = input.permute(0, 2, 1)
        elif dim == 0:
            input = input.permute(1, 2, 0)

        b = input.shape[0]
        batch_list = []
        for idx in range(b):
            inp = input[idx]
            sample = torch.multinomial(inp, num_samples, replacement=replacement)
            batch_list.append(sample)

        input = torch.stack(batch_list)

        if dim == 1:
            input = input.permute(0, 2, 1)
        elif dim == 0:
            input = input.permute(2, 0, 1)

    return input

