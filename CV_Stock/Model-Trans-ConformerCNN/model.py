import torch
import torch.nn as nn
from torchaudio.models import Conformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conformer_CNN(nn.Module):
    def __init__(self, num_class, conformer = False, res = True):
        super(Conformer_CNN, self).__init__()

        """self.conv_init1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_init2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_init3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_init4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_init5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_init6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.maxpool_u = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)"""


        # =======
        # Unet
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, num_class)
        self.ln1 = nn.LayerNorm((5, 100, 100))

        # =======
        # Conformer
        self.conformer = Conformer(
            input_dim=100,
            num_heads=5,
            ffn_dim=128,
            num_layers=16,
            depthwise_conv_kernel_size=31)



    def MinMax(self, x):
        max_ = x.max().item()
        min_ = x.min().item()
        x = (x - min_) / (max_ - min_)
        return x

    def forward(self, x):
        """
        x: batch, 1, 224, 224
        """
        # =======
        # CNN
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        """

        # =======
        # Conformer
        """
        Input scale: (0, 255)
        Output scale: (0, 255)
        """
        x_i = x.clone()
        x_s = x.size()
        x = x.view(x_s[0], x_s[1] * x_s[3], x_s[2])
        lengths = torch.tensor([x.shape[1] for i in range(len(x))]).to(device)
        x, len_ = self.conformer(x, lengths)
        x = x.view(x_s)
        x = x + x_i
        x = self.ln1(x)

        # =======
        # CNN
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpool(self.relu(self.bn3(self.conv3(x))))
        x = self.maxpool(self.relu(self.bn4(self.conv4(x))))
        x = self.maxpool(self.relu(self.bn5(self.conv5(x))))
        
        # print(x.size())
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # l = self.softmax(l)

        return x