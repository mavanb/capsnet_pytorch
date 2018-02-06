import torch
from torch.autograd import Variable
import torch.nn as nn


def variable(tensor, volatile=False):
    if torch.cuda.is_available():
        return Variable(tensor, volatile=volatile).cuda()
    return Variable(tensor, volatile=volatile)


def parameter(tensor):
    if torch.cuda.is_available():
        return nn.Parameter(tensor).cuda()
    return nn.Parameter(tensor)


def new_grid_size(old_height, old_width, stride, kernel_size, padding):
    """ Calculate new images size after convoling.
    Used formula from: https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-
    Networks-Part-2/
    """
    def calc(x): return int((x - kernel_size + 2 * padding)/stride + 1)
    return calc(old_height), calc(old_width)


def squash(tensor, dim=-1):
    #todo check if safe norm is required here: not present in pytorch githubs
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
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


def dynamic_routing(u_hat, iters=3):
    """ Implementation of routing algorithm described in Dynamic Routing Hinton 2017.

    :param u_hat: Variable containing the of the next layer capsules given each previous layer capsule. shape:
    [batch_size, num_caps_next_layer, num_caps_prev_layer, dim_next_layer]
    :param iters: number of iterations in routing algo.
    :return: next layer predictions sized by probability/correspondence. shape: [batch_size, num_caps_next_layer,
    dim_next_layer]
    """
    b, j, i, n = u_hat.shape
    b_vec = variable(torch.zeros(b, i, j))
    for _ in range(iters):
        # softmax of j
        c_vec = torch.nn.Softmax(dim=2)(b_vec)

        # in einsum: "bij, bjin-> bjn"
        s_vec = torch.matmul(c_vec.view(b, j, 1, i), u_hat).squeeze()
        v_vec = squash(s_vec)

        # in einsum: "bjin, bjn-> bij", inner product over n,
        # note: use x=x+1 instead of x+=1 to ensure new object creation and avoid inplace operation
        b_vec = b_vec + torch.matmul(u_hat.view(b, j, i, 1, n), v_vec.view(b, j, 1, n, 1)).squeeze().permute(0, 2, 1)
    return v_vec



