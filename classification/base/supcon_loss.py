"""
Supervised Contrastive Loss (SupCon)

Reference: "Supervised Contrastive Learning" (NeurIPS 2020, Khosla et al.)

Applied to EEG emotion recognition: pulls together feature representations
of same-emotion samples while pushing apart different-emotion samples,
learning subject-invariant emotional representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss.

    Args:
        temperature: scalar temperature for scaling similarities (default: 0.07)
        base_temperature: base temperature for normalization (default: 0.07)
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: hidden vectors of shape [batch_size, feature_dim]
            labels: ground truth labels of shape [batch_size]

        Returns:
            scalar loss
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return features.new_tensor(0.0)

        # Normalize features to unit hypersphere
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs (same label, excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Exclude self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Check if there are any positive pairs
        positives_per_row = mask.sum(1)
        valid_rows = positives_per_row > 0
        if not valid_rows.any():
            return features.new_tensor(0.0)

        # For numerical stability
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (positives_per_row + 1e-12)

        # Loss (only for rows with positive pairs)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[valid_rows].mean()

        return loss


class CenterLoss(nn.Module):
    """Center Loss for learning discriminative features.

    Reference: "A Discriminative Feature Learning Approach for Deep Face Recognition"
    (ECCV 2016, Wen et al.)

    Learns a center for each class and penalizes the distance between features
    and their corresponding class centers.

    Args:
        num_classes: number of classes
        feat_dim: feature dimension
    """

    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feat_dim]
            labels: [batch_size]
        Returns:
            center loss scalar
        """
        batch_centers = self.centers[labels]
        loss = ((features - batch_centers) ** 2).sum(dim=1).mean()
        return loss
