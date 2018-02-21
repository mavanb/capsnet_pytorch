import torch
from torch.autograd import Variable
import torch.nn as nn
import math

def variable(tensor, volatile=False):
    if torch.cuda.is_available():
        return Variable(tensor, volatile=volatile).cuda()
    return Variable(tensor, volatile=volatile)


def parameter(tensor):
    if torch.cuda.is_available():
        return nn.Parameter(tensor).cuda()
    return nn.Parameter(tensor)


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
    return scale * tensor / torch.sqrt(squared_norm)


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
            return torch.sparse.torch.eye(depth).cuda().index_select(dim=0, index=labels)
        else:
            return torch.sparse.torch.eye(depth).index_select(dim=0, index=labels)

# c_vec_temp = variable(torch.FloatTensor(128, 10, 1152).fill_(8.6806 / 10000))


def dynamic_routing(u_hat, iters):
    """
    Implementation of routing algorithm described in Dynamic Routing Hinton 2017.
    :param input: u_hat, Variable containing the of the next layer capsules given each previous layer capsule. shape:
    [batch_size, num_caps_next_layer, num_caps_prev_layer, dim_next_layer]
    :param iters: number of iterations in routing algo.
    :return: next layer predictions sized by probability/correspondence. shape: [batch_size, num_caps_next_layer,
    dim_next_layer]
    """
    b, j, i, n = u_hat.shape
    b_vec = variable(torch.zeros(b, j, i))

    for index in range(iters):
        # softmax of i, weight of all predictions should sum to 1, note in tf code this does not give an error
        c_vec = torch.nn.Softmax(dim=2)(b_vec)

        # in einsum: bij, bjin-> bjn
        # in matmul: bj1i, bjin = bj (1i)(in) -> bjn
        s_vec = torch.matmul(c_vec.view(b, j, 1, i), u_hat).squeeze()
        v_vec = squash(s_vec)

        if index < (iters - 1):  # skip update last iter
            # in einsum: "bjin, bjn-> bij", inner product over n
            # in matmul: bji1n, bj1n1 = bji (1n)(n1) = bji1
            # note: use x=x+1 instead of x+=1 to ensure new object creation and avoid inplace operation
            b_vec = b_vec + torch.matmul(u_hat.view(b, j, i, 1, n), v_vec.view(b, j, 1, n, 1)).squeeze()

    return v_vec




