from torchvision import transforms

import torch
import numpy as np
from data.data_loader import get_dataset
from utils import variable
from nets import BasicCapsNet

def convert_grid2flat(tot_caps=32, tot_height=6, tot_width=6):
    def convert(height, width):
        return [caps_idx * tot_height * tot_width + height * tot_width + width for caps_idx in range(tot_caps)]
    return convert


# load model, allow GPU trained models
model =  torch.load("./trained_models/simple_caps_net_cuda", map_location=lambda storage, loc: storage)
model.eval()

grid_height = 6
grid_width = 6
prim_capsules = 32
labels = 10

histograms = np.zeros(shape=(labels, prim_capsules))

transform = transforms.ToTensor()
dataset, data_shape, label_shape = get_dataset("mnist", transform=transform, train=True)
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
test_loader = torch.utils.data.DataLoader(dataset, batch_size=128, drop_last=True, **kwargs, shuffle=True)

for batch_idx, (data, labels) in enumerate(test_loader):

    def log_function(iter_index, u_hat, b_vec, c_vec, v_vec, s_vec, s_vec_bias):
        if iter_index == 2:
            c_vec = c_vec.data.cpu().numpy()
            caps_indices = convert_grid2flat()(1, 1)
            max_indices = c_vec.take(caps_indices, axis=2).argmax(axis=2)

            # loop over the 128 argmaxs (128,10).
                # check label in labels
                # add each of the arg max indeces to the histogram
    model.dynamic_routing.log_function = log_function

    _ = model(variable(data))


