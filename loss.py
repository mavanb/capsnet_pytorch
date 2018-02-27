from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from utils import one_hot


class CapsuleLoss(_Loss):

    def __init__(self, m_plus, m_min, alpha, num_classes, size_average=True):
        super(CapsuleLoss, self).__init__(size_average)

        self.m_plus = m_plus
        self.m_min = m_min
        self.num_classes = num_classes
        self.alpha = alpha

        self.reconstruction_loss = nn.MSELoss(reduce=False)

    def forward(self, images, labels, class_probs, reconstructions):
        labels_one_hot = one_hot(labels, self.num_classes)
        present_loss = 0.5 * F.relu(self.m_plus - class_probs, inplace=True) ** 2
        absent_loss = 0.5 * F.relu(class_probs - self.m_min, inplace=True) ** 2

        margin_loss = labels_one_hot * present_loss + 0.5 * (1. - labels_one_hot) * absent_loss
        margin_loss_per_sample = margin_loss.sum(dim=1)
        margin_loss = margin_loss_per_sample.mean() if self.size_average else margin_loss_per_sample.sum()
        reconstruction_loss = self.reconstruction_loss(reconstructions, images).sum(dim=-1)
        if self.size_average:
            reconstruction_loss = reconstruction_loss.mean()
        else:
            reconstruction_loss = reconstruction_loss.sum()

        total_loss = margin_loss + self.alpha * reconstruction_loss
        return total_loss, margin_loss, reconstruction_loss * self.alpha

