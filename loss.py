from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from utils import one_hot


class CapsuleLoss(_Loss):

    def __init__(self, m_plus, m_min, alpha, num_classes, include_recon, size_average=True):
        super(CapsuleLoss, self).__init__(size_average)

        self.m_plus = m_plus
        self.m_min = m_min
        self.num_classes = num_classes
        self.alpha = alpha
        self.include_recon = include_recon

        self.recon_loss = nn.MSELoss(reduce=False)

    def forward(self, images, labels, logits, reconstructions):
        labels_one_hot = one_hot(labels, self.num_classes)
        present_loss = 0.5 * F.relu(self.m_plus - logits, inplace=True) ** 2
        absent_loss = 0.5 * F.relu(logits - self.m_min, inplace=True) ** 2

        margin_loss = labels_one_hot * present_loss + 0.5 * (1. - labels_one_hot) * absent_loss
        margin_loss_per_sample = margin_loss.sum(dim=1)
        margin_loss = margin_loss_per_sample.mean() if self.size_average else margin_loss_per_sample.sum()

        if self.include_recon:
            recon_loss = self.recon_loss(reconstructions, images).sum(dim=-1)
            if self.size_average:
                recon_loss = recon_loss.mean()
            else:
                recon_loss = recon_loss.sum()
        else:
            recon_loss = None

        if self.include_recon:
            total_loss = margin_loss + self.alpha * recon_loss
        else:
            total_loss = margin_loss

        return total_loss, margin_loss, recon_loss

