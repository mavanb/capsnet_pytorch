from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from utils import one_hot


class CapsuleLoss(_Loss):

    def __init__(self, m_plus, m_min, alpha, beta, num_classes, include_recon, include_entropy, caps_sizes=None,
                 size_average=True):
        super(CapsuleLoss, self).__init__(size_average)

        self.m_plus = m_plus
        self.m_min = m_min
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.include_recon = include_recon
        self.include_entropy = include_entropy

        self.caps_sizes = caps_sizes

        self.recon_loss = nn.MSELoss(reduce=False)

    def forward(self, images, labels, logits, recon, entropy):
        labels_one_hot = one_hot(labels, self.num_classes)
        present_loss = 0.5 * F.relu(self.m_plus - logits, inplace=True) ** 2
        absent_loss = 0.5 * F.relu(logits - self.m_min, inplace=True) ** 2

        margin_loss = labels_one_hot * present_loss + 0.5 * (1. - labels_one_hot) * absent_loss
        margin_loss_per_sample = margin_loss.sum(dim=1)
        margin_loss = margin_loss_per_sample.mean() if self.size_average else margin_loss_per_sample.sum()

        if self.include_recon:
            recon_loss = self.recon_loss(recon, images).sum(dim=-1)
            if self.size_average:
                recon_loss = recon_loss.mean()
            else:
                recon_loss = recon_loss.sum()
        else:
            recon_loss = None

        if self.include_entropy:
            # select entropy of last layer
            last_iter_entropy = entropy[:, :, -1]

            # eliminate the batch dimension
            if self.size_average:
                last_iter_entropy = last_iter_entropy.mean(dim=1)
            else:
                last_iter_entropy = last_iter_entropy.sum(dim=1)

            # multiply the entropy of each layer by its size
            caps_sizes = self.caps_sizes.float()
            caps_sizes_weights = caps_sizes / sum(caps_sizes)

            entropy_loss = (last_iter_entropy * caps_sizes_weights).sum()

        else:
            entropy_loss = None

        total_loss = margin_loss

        if self.include_recon:
            total_loss = total_loss + self.alpha * recon_loss
        if self.include_entropy:
            total_loss = total_loss + self.beta * entropy_loss

        return total_loss, margin_loss, recon_loss, entropy_loss

