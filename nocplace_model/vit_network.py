import torch
from torch import nn

from nocplace_model.layers import Flatten, L2Norm, GeM
import nocplace_model.salad_layer as Salad


CHANNELS_NUM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}

class DINOv2(nn.Module):

    def __init__(self, backbone : str, num_trainable_blocks = 4, norm_layer = True, return_token = True):
        super().__init__()
        assert backbone in CHANNELS_NUM, f"backbone must be one of {list(CHANNELS_NUM.keys())}"
        self.model = torch.hub.load('facebookresearch/dinov2', backbone)
        self.channels_num = CHANNELS_NUM[backbone]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):

        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)

        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        f = f.reshape((B, H // 14, W // 14, self.channels_num)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f

class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone : str):
        super().__init__()
        self.backbone = DINOv2(backbone = backbone)
        self.channels_num = CHANNELS_NUM[backbone]
        self.aggregator = Salad.SALAD()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x