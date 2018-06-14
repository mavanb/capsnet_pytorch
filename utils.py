import torch
import math
import logging
import sys


def flex_profile(func):
    """ Decorator for the @proflile decorator of kernprof. Avoids having to remove it all the time.

    To profile run: kernprof -v -l train_capsnet.py

    Make sure to first put @profile (should be removed afterwards) or @flex_profile (no need to remove) decorators above
    the functions you want to profile.

    Note: I profilled the effect of this decorator itself: decorator is only called on function init. Thus, only do not
    use in frequently called nested functions (in others words, if the function is instantiated very often.)
    """
    try:
        func = profile(func)
    except NameError:
        pass
    return func


def new_grid_size(grid, kernel_size, stride=1, padding=0):
    """ Calculate new images size after convoling.
    Used formula from: https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-
    Networks-Part-2/
    """
    def calc(x): return int((x - kernel_size + 2 * padding)/stride + 1)
    return calc(grid[0]), calc(grid[1])


def padding_same_tf(grid, kernel, stride):
    """ TensorFlow padding SAME corresponds to padding such that: output size = ceil(input/stride)"""
    def calc(x):
        out_size = math.ceil(x / stride)
        return int(((out_size - 1) * stride + kernel - x) / 2)
    return calc(grid[0]), calc(grid[1])


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1. + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-7)


def calc_entropy(input_tensor, dim):
    """ Calculate entropy over dimension of tensor. The sum over this dimension should thus always sum to one.

    Args:
        input_tensor: (tensor) Input tensor.
        dim: Dimension over which the entropy is calculated. All entries over this dimension should sum to 1.

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
        input_tensor: (tensor) tensor to compare against of any size
        value: (float) value to compare to
        precision: (float) precision of comparision

    Returns: (tensor) bool tensor of same size as input indication which entries are approx equal

    """
    return (input_tensor * precision).round() == value * precision


def one_hot(labels, depth):
    """ Create one-hot encoding matrix from vector of labels/indices.
    :param labels: 1D-tensor or 1D-Variable
    :param depth: output length of one hot vectors i.e. number of classes
    :return: 2D-tensor or 2D-Variable (depending on input) of shape [len(labels), depth]
    # """
    return torch.eye(depth, device=get_device()).index_select(dim=0, index=labels)


def init_weights(module, weight_mean=0, weight_stddev=0.1, bias_mean=0.1):
    module.weight.data.normal_(weight_mean, weight_stddev)
    module.bias.data.fill_(bias_mean)
    return module


def convert_grid_index_to_flat(tot_caps, tot_height, tot_width):
    """ Given grid size and number of channels returns function that gives all indices that come from a certain postion
    in the old grid."""
    def convert(height, width):
        """ Height, width are the ocation of in the grid, returns the indices in the flattened vector."""
        return [caps_idx * tot_height * tot_width + height * tot_width + width for caps_idx in range(tot_caps)]
    return convert


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name):
        logger = logging.getLogger(name)
        handler = logging.StreamHandler(sys.stdout)
        # formatter = logging.Formatter(
        #     '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        # handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger


def batched_index_select(input, dim, index):
    """ Select indices batch_wise.
    Resources:

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



