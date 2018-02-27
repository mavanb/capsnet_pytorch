""" Train, Validation and Test Split for torchvision Datasets

Code based on this example (https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb). Generalized to handle
multiple data sets.
"""
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from torchvision.datasets import MNIST, CIFAR10
import torch.utils.data as data
from torchvision.transforms import ToTensor
from utils import one_hot, variable


def get_train_valid_data(data_set, batch_size, seed=None, valid_size=0.1, shuffle=True, num_workers=4,
                         pin_memory=False, train_max=None, valid_max=None, drop_last=False):
    """
    Utility function for loading and returning train, valid and test data.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_set: instance of torch Dataset class
    - batch_size: how many samples per batch to load.
    - seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - train_max: Maximum number of samples in train set, mostly for debugging purposes
    - valid_max:  .. in valid set, ..

    Returns: train and valid dataloader
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(np.floor((1-valid_size) * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[:split], indices[split:]

    # limit number of train / valid samples
    if train_max:
        assert (train_max * batch_size < len(train_idx)), "train_max should be lower than number of samples in train set"
        train_idx = train_idx[:train_max * batch_size]
    if valid_max:
        assert (valid_max * batch_size < len(valid_idx)), "valid_max should be lower than number of samples in valid set"
        valid_idx = valid_idx[:valid_max * batch_size]

    if shuffle:
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
    else:
        train_sampler = SequentialSampler(train_idx)
        valid_sampler = SequentialSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(data_set,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    valid_loader = torch.utils.data.DataLoader(data_set,
                                               batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    return train_loader, valid_loader


def get_dataset(dataset_name, transform=ToTensor()):
    if dataset_name == "mnist":
        dataset = MNIST(download=True, root="./mnist", transform=transform, train=True)
    elif dataset_name == "cifar10":
        dataset = CIFAR10(download=True, root="./cifar10", transform=transform, train=True)
    else:
        raise ValueError("Name dataset does not exists.")
    return dataset, dataset[0][0].shape


class Gaussian2D(data.Dataset):

    def __init__(self, new_dim=5, n_samples=(5000, 2000), mix_coef=0.5, covs=None, means=None, train=True,
                 transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        if covs is None:
            cov = np.diag([1, 1])
            covs = (cov, cov)
        else:
            assert covs[0].shape == covs[1].shape == (2, 2), "covs should be tuple of 2 2x2 numpy arrays"

        if means is None:
            means = ([-1, -1], [1, 1])
        else:
            assert means.shape == (2, 2), "means should be 2x2 numpy array"

        self.means = means
        self.covs = covs
        self.mix_coef = mix_coef
        self.new_dim = new_dim

        self.data_observed, self.data_latent, self.labels = self.create_samples_two2d_gaussian(n_samples[0] if train
                                                                                            else n_samples[1])

    def create_samples_two2d_gaussian(self, n_samples):
        labels = np.random.binomial(1, p=self.mix_coef, size=n_samples)
        labels_one_hot = one_hot(torch.from_numpy(labels), 2)

        gaus_1 = np.random.multivariate_normal(mean=self.means[0], cov=self.covs[0], size=n_samples)
        gaus_2 = np.random.multivariate_normal(mean=self.means[1], cov=self.covs[1], size=n_samples)
        two2d_gaussian = np.stack([gaus_1, gaus_2], axis=0)
        latent_samples = np.einsum("nm,mnj->nj", labels_one_hot, two2d_gaussian)
        observed_samples = self.transform_latent_data(latent_samples)
        return observed_samples, latent_samples, labels

    def transform_latent_data(self, latent_samples):
        W1 = np.random.rand(5, latent_samples.shape[1])
        trans1 = np.sin(np.einsum("ij,kj->ki", W1, latent_samples))
        W2 = np.random.rand(self.new_dim, 5)
        trans2 = np.einsum("ij,kj->ki", W2, trans1)
        return trans2

    def __len__(self):
        return len(self.data_observed)

    def __getitem__(self, item):
        observed = self.data_observed[item]
        latent = self.data_latent[item]
        label = self.labels[item]

        if self.transform is not None:
            observed = self.transform(observed)
            latent = self.transform(latent)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return observed, label, latent






