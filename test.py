# #
# # #
# # # import torch
# # #
# # # def to_sparse(x):
# # #     """ converts dense tensor x to sparse format """
# # #     x_typename = torch.typename(x).split('.')[-1]
# # #     sparse_tensortype = getattr(torch.sparse, x_typename)
# # #
# # #     indices = torch.nonzero(x)
# # #     if len(indices.shape) == 0:  # if all elements are zeros
# # #         return sparse_tensortype(*x.shape)
# # #     indices = indices.t()
# # #     values = x[tuple(indices[i] for i in range(indices.shape[0]))]
# # #     return sparse_tensortype(indices, values, x.size())
# # #
# # # a = torch.randn(100,100)
# # # b = torch.randn(100,100)
# #
# # import torch
# #
# # x = torch.randn(3, 4)
# #
# # sorted, indices = torch.sort(x)
# #
# # pass
#
# # import torch
# # print(torch.__version__)
# # x = torch.randn(3, 4)
# # indices = torch.tensor([0, 2])
# # t = torch.index_select(x, 0, indices)
# #
# # pass
#
# import torch
# import time
# from utils import get_device
# device = get_device()
#
# indices = torch.tensor([
#             [0, 2],
#             [1, 2],
#             [0, 1]
#         ], device=device)
#
# bool_indices = torch.ByteTensor([
#         [1, 0, 1],
#         [0, 1, 1],
#         [1, 1, 0]
# ], device=device)
#
# x = torch.tensor([
#         [1, 2, 3],
#         [4, 5, 6],
#         [6, 7, 8]
#     ], device=device)
#
# from utils import batched_index_select
#
# def index_select_fast(input, indices):
#     rows, cols = input.shape
#     flat_indices = (indices + (torch.tensor([range(rows)], device=device) * cols).view(-1, 1)).view(-1)
#     new = torch.index_select(input.view(-1), 0, flat_indices).view(indices.shape)
#     return new
#
#
# def index_select_2d_loop(input, indices):
#     new = torch.stack([input[i, indices[i]] for i in range(input.shape[0])])
#     return new
#
#
# print(batched_index_select(x, 1, indices))
# print(x[bool_indices].view(-1, 2))
# print(index_select_2d_loop(x, indices))
# print(index_select_fast(x, indices))
#
# # speed check
# num_col = 1000
# num_row = 5000
# indices_speed_check = torch.randint(0, num_col, (num_row, 2), dtype=torch.long, device=device)
# x_speed_check = torch.randn(num_row, num_col, device=device)
#
# bool_indices_speed_check = torch.zeros(5000, 1000, device=device)
# bool_indices_speed_check[:, 0:2] = 1
# bool_indices_speed_check = bool_indices_speed_check.to(torch.uint8)
#
# start = time.time()
# x_speed_check[bool_indices_speed_check].view(-1, 2)
# run_time = (time.time() - start) * 1000
# print("index method: {}".format(run_time))
#
# start = time.time()
# index_select_fast(x_speed_check, indices_speed_check)
# run_time = (time.time() - start) * 1000
# print("index_select method: {}".format(run_time))
#
# start = time.time()
# idxs = bool_indices_speed_check.nonzero()[:,1].view(5000, -1)
# batched_index_select(x_speed_check, 1, idxs)
# run_time = (time.time() - start) * 1000
# print("batched_index_select from mask: {}".format(run_time))
#
# start = time.time()
# batched_index_select(x_speed_check, 1, indices_speed_check)
# run_time = (time.time() - start) * 100
# print("batched_index_select from indices: {}".format(run_time))
#
# # requires loop
# start = time.time()
# index_select_2d_loop(x_speed_check, indices_speed_check)
# run_time = (time.time() - start) * 1000
# print("index_select loop: {}".format(run_time))
#

import time
import torch
from utils import get_device

start = time.time()
b = None
for _ in range(2*421):
    if b is None:
        b = torch.zeros(128, 10, 1152, device=get_device(),  requires_grad=False)
    c = b
print((time.time() - start)/(421*128))