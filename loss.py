""" Loss module.

References:
    [1] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing between capsules,” in NIPS, pp. 3859–3869, 2017.

"""
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from utils import one_hot


class CapsuleLoss(_Loss):
    """ Margin Loss

    Margin Loss as defined in [1]. Additionally, an loss term is added: the average routing entropy.

    Args:
        m_plus (float): m+ in the margin loss.
        m_min (float): m- in the margin loss.
        alpha (float): the scalar that controls the contribution of the reconstruction loss.
        beta (float): the scalar that controls the contribution of the entropy penalty.
        include_recon (bool): Use the reconstruction loss.
        include_entropy (bool): Use the additional entropy penalty.
        caps_sizes (LongTensor, optional): Specifies the number of capsules in the non-primary capsule layers. Default
            None, but should be given in include_entropy. Shape: num layers.
    """

    def __init__(self, m_plus, m_min, alpha, beta, include_recon, include_entropy, caps_sizes=None):
        super(CapsuleLoss, self).__init__()

        self.m_plus = m_plus
        self.m_min = m_min
        self.alpha = alpha
        self.beta = beta
        self.include_recon = include_recon
        self.include_entropy = include_entropy

        self.caps_sizes = caps_sizes

        # init mean square error loss
        self.recon_loss = nn.MSELoss(reduction="none")

    def forward(self, images, labels, logits, recon, entropy=None):
        """ Forward pass.

        Args:
            images (FloatTensor): Orginal images. Shape: [batch, channel, height, width].
            labels (LongTensor): Class labels. Shape: [batch]
            logits (FloatTensor): Class logits. Length of the final capsules. Shape: [batch, classes]
            recon (FloatTensor): Reconstructed image. Same shape as images.
            entropy (FloatTensor, optional): Average Entropy per layer per routing iter. Shape: [layer, batch,
                rout_iter].

        Returns:
            total_loss (FloatTensor): Sum of all losses. Single value.
            margin_loss (FloatTensor): Margin loss defined in [1]. Single value.
            recon_loss (FloatTensor): MSE loss of the reconstructed image. None if not included. Single value.
            entropy_loss (FloatTensor): Entropy loss determined by the average entropy in the last routing iter. None if
                not included. Single value.

        """
        num_classes = logits.shape[1]
        labels_one_hot = one_hot(labels, num_classes)

        # the factor 0.5 in front of both terms is not in the paper, but used in the source code
        present_loss = 0.5 * F.relu(self.m_plus - logits, inplace=True) ** 2
        absent_loss = 0.5 * F.relu(logits - self.m_min, inplace=True) ** 2

        # the factor 0.5 is the downweight mentioned in the Margin loss in [1]
        margin_loss = labels_one_hot * present_loss + 0.5 * (1. - labels_one_hot) * absent_loss
        margin_loss_per_sample = margin_loss.sum(dim=1)
        margin_loss = margin_loss_per_sample.mean()

        if self.include_recon:

            # sum over all image dimensions
            recon_loss = self.recon_loss(recon, images).sum(dim=-1).sum(dim=-1).sum(dim=-1)
            assert len(recon_loss.shape) == 1, "Only batch dimension should be left after in recon loss."

            # average of sum over batch dimension
            recon_loss = recon_loss.mean()
        else:
            recon_loss = None

        if self.include_entropy:
            if self.caps_sizes is None:
                raise ValueError("If include entropy, the size of each capsule layer should be known.")

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

        # scale the recon and entropy loss by the scaling factors
        if self.include_recon:
            total_loss = total_loss + self.alpha * recon_loss
        if self.include_entropy:
            total_loss = total_loss + self.beta * entropy_loss

        return total_loss, margin_loss, recon_loss, entropy_loss

