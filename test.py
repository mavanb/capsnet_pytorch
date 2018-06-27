import torch
import time
from utils import get_device, multinomial_3d

# input = torch.tensor(
#     [
#         [0.98, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
#         [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.98]
#     ]
#
# )



b = 128
input = torch.rand(b, 10, 1152, requires_grad=False, device=get_device())
weights = torch.rand(b, 10, 1152, requires_grad=False, device=get_device())

start = time.time()

indices = multinomial_3d(weights, 4, dim=1)

x = input.scatter_(1, indices, 0)

comp_time = time.time() - start
print(comp_time * 1000 / 128)

