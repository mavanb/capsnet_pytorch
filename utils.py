import torch
from torch.autograd import Variable
import torch.nn as nn
import math


def flex_profile(func):
    """ Decorator for the @proflile decorator of kernprof. Avoids having to remove it all the time. Profiled the effect
     of this decorator itself: decorator is only called on function init. Note: do not use in frequently called nested
    functions (in others words, if the function is instantiated very often."""
    try:
        func = profile(func)
    except NameError:
        pass
    return func


def variable(tensor, volatile=False):
    if torch.cuda.is_available():
        return Variable(tensor, volatile=volatile).cuda()
    return Variable(tensor, volatile=volatile)


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
    #todo check if safe norm is required here: not present in pytorch githubs
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1. + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-7)


def one_hot(labels, depth):
    """ Create one-hot encoding matrix from vector of labels/indices.
    :param labels: 1D-tensor or 1D-Variable
    :param depth: output length of one hot vectors i.e. number of classes
    :return: 2D-tensor or 2D-Variable (depending on input) of shape [len(labels), depth]
    """
    if type(labels) == Variable:
        return variable(torch.sparse.torch.eye(depth)).index_select(dim=0, index=labels)
    else:
        if torch.cuda.is_available():
            return torch.sparse.torch.eye(depth).cuda().index_select(dim=0, index=labels.cuda())
        else:
            return torch.sparse.torch.eye(depth).index_select(dim=0, index=labels)


def init_weights(module, weight_mean=0, weight_stddev=0.1, bias_mean=0.1):
    module.weight.data.normal_(weight_mean, weight_stddev)
    module.bias.data.fill_(bias_mean)
    return module



