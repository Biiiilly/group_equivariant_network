import torch
import torch.nn as nn
import torch.nn.functional as F
from gcnn_p4.layer_p4 import Z2P4GConv2d
from gcnn_p4.max_pool_p4 import GConv2d_MaxPooling

# This is the model that we are going to train:
class P4GConvNet(nn.Module):

    def __init__(self):

        super(P4GConvNet, self).__init__()
        # self.batck_size = batch_size
        self.gconv1 = Z2P4GConv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.gconv2 = Z2P4GConv2d(in_channels=20, out_channels=20, kernel_size=3)
        self.fc     = nn.Linear(20 * 7 * 7, 10)

    def forward(self, x):

        x = F.relu(self.gconv1(x))
        x = GConv2d_MaxPooling(x, 2)

        x = F.relu(self.gconv2(x))
        x = GConv2d_MaxPooling(x, 2)

        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        x = torch.sort(x, dim=2)[0]
        x = x.view(x.size(0), -1)

        x = F.dropout(x, training=self.training, p=0.1)
        x = self.fc(x)

        return F.log_softmax(x)
    

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()
        # self.batck_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, padding=1)
        self.fc     = nn.Linear(20 * 7 * 7, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training, p=0.1)

        x = self.fc(x)

        return F.log_softmax(x)
