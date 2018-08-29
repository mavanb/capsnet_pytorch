import numpy as np
import torch.utils.data as data
from utils import one_hot
import torch


class Gaussian2D(data.Dataset):

    def __init__(self, new_dim=6, n_samples=(5000, 2000), mix_coef=0.5, covs=None, means=None, train=True,
                 transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        if covs is None:
            cov = np.diag([0.5, 0.5])
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

        clear0 = np.all(np.stack([self.data_latent.sum(axis=1)<0, self.labels==0], axis=1), axis=1).sum()
        clear1 = np.all(np.stack([self.data_latent.sum(axis=1)>0, self.labels == 1], axis=1), axis=1).sum()
        prac0 = np.all(np.stack([self.data_latent.sum(axis=1)<0.25*np.sqrt(2), self.labels==0], axis=1), axis=1).sum()
        prac1 = np.all(np.stack([self.data_latent.sum(axis=1)>0.25*np.sqrt(2), self.labels==1], axis=1), axis=1).sum()
        self.theoretical_acc = (clear0 + clear1) / (n_samples[0] if train else n_samples[1])
        self.practical_acc = (prac0 + prac1) / (n_samples[0] if train else n_samples[1])

        print("{} set, Theoretical accuracy: {}, Practical accuracy: {}".format("Training" if train else "Validate",
                                                                        self.theoretical_acc, self.practical_acc))

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
        W1 = np.random.rand(self.new_dim, latent_samples.shape[1])
        trans = np.sin(np.einsum("ij,kj->ki", W1, latent_samples))
        W2 = np.random.rand(self.new_dim, self.new_dim)
        trans= np.einsum("ij,kj->ki", W2, trans)
        return trans

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



