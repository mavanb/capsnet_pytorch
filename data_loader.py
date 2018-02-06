""" Train, Validation and Test Split for torchvision Datasets

Code based on this example (https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb). Generalized to handle
multiple data sets.
"""
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_test(data_set, batch_size, random_seed, valid_size=0.1, shuffle=True, num_workers=4,
                         pin_memory=False):
    """
    Utility function for loading and returning train, valid and test data.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_set: instance of torch Dataset class
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns: train and valid dataloader
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(data_set,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(data_set,
                                               batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader


