"""
HARBench Classifiers

Classification heads for Human Activity Recognition.
"""

import torch
import torch.nn as nn


class TwoLayerClassifier(nn.Module):
    """
    Two-layer classifier for HAR.

    Structure: Linear -> ReLU -> Linear

    Used for fine-tuning pretrained backbones.
    """

    def __init__(self, backbone: nn.Module, n_classes: int, hidden_dim: int = 512):
        """
        Args:
            backbone: Feature extractor backbone (must have output_dim attribute)
            n_classes: Number of output classes
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.backbone = backbone
        input_dim = backbone.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_classes, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, channels, sequence_length)

        Returns:
            Logits (batch_size, n_classes)
        """
        # Extract features
        features = self.backbone(x)

        # Flatten if needed (e.g., from (batch, 512, 1) to (batch, 512))
        features = features.reshape(features.size(0), -1)

        # Classify
        return self.classifier(features)


class ThreeLayerClassifier(nn.Module):
    """
    Three-layer classifier with dropout and batch normalization.

    Structure: Linear -> ReLU -> Dropout -> BN -> Linear -> ReLU -> Dropout -> BN -> Linear

    More robust classifier for complex tasks.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
    ):
        """
        Args:
            backbone: Feature extractor backbone
            n_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.backbone = backbone
        input_dim = backbone.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, n_classes, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, channels, sequence_length)

        Returns:
            Logits (batch_size, n_classes)
        """
        features = self.backbone(x)
        features = features.reshape(features.size(0), -1)
        return self.classifier(features)
