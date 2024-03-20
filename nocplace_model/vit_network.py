import torch
from torch import nn

from nocplace_model.layers import Flatten, L2Norm, GeM


CHANNELS_NUM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone : str, fc_output_dim : int):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str)
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
            train_all_layers (bool): whether to freeze the first layers of the backbone during training or not.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM, f"backbone must be one of {list(CHANNELS_NUM.keys())}"
        self.features_dim = CHANNELS_NUM[backbone]
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(self.features_dim, fc_output_dim),
            L2Norm()
        )
    
    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = L2Norm()(x)  # Normalize the features
        x = self.aggregation[3:](x)
        return x