import torch
import torchvision
import torch.nn as nn

# Vision Transformer: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
# try pretrain, same link above
class VT_CNN(nn.Module):
    def __init__(self):
        super(VT_CNN, self).__init__()

        # =======
        # Unet
        self.conv_init = nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.VisionTransformer = torchvision.models.vit_b_16(weights='DEFAULT')
        self.ln_init = nn.LayerNorm(1000)
        self.fc = nn.Linear(1000, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Input size: 1, 5, 100, 100
        Input scale: (0, 255)
        Output scale: (0, 255)
        """
       
        x = self.conv_init(x)
        x = self.VisionTransformer(x)
        x = self.ln_init(x)
        x = self.relu(x)        
        x = self.fc(x)

        return x