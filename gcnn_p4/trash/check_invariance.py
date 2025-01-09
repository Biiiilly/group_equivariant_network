import torch
import torch.nn as nn
import torch.nn.functional as F
import P4GConv2d as g


def check_invariance(input_type, in_channels, out_channels, output_features, kernel_size, batch_size, image_size, Z2tolerance, P4tolerance):

    if input_type == 'Z2':

        x = torch.randn(batch_size, in_channels, image_size, image_size)
        network = Z2_lastlayer(image_size=image_size, in_channels=in_channels, out_channels=out_channels, output_features=output_features, kernel_size=kernel_size)
        result = network(x)
        #print(result)

        for i in range(1, 4):
            x_rot = torch.rot90(x, 1, (2, 3))
            result_rot = network(x_rot)
            if torch.allclose(result, result_rot, atol=Z2tolerance) == False:
                print(result)
                print(result_rot)
                print(torch.norm(result-result_rot))
            assert torch.allclose(result, result_rot, atol=Z2tolerance)

    elif input_type == 'P4':
        
        x = torch.randn(batch_size, in_channels, 4, image_size, image_size)
        network = P4_lastlayer(image_size=image_size, in_channels=in_channels, out_channels=out_channels, output_features=output_features, kernel_size=kernel_size)
        result = network(x)

        for i in range(1, 4):
            x_rot = torch.rot90(x, 1, (3, 4))
            result_rot = network(x_rot)
            if torch.allclose(result, result_rot, atol=P4tolerance) == False:
                print(result)
                print(result_rot)
                print(torch.norm(result-result_rot))
            assert torch.allclose(result, result_rot, atol=P4tolerance)


class Z2_lastlayer(nn.Module):

    def __init__(self, image_size, in_channels, out_channels, output_features, kernel_size):

        super(Z2_lastlayer, self).__init__()
        self.gconv1 = g.Z2P4GConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.fc = nn.Linear(out_channels * int(image_size/2) * int(image_size/2), output_features)
    
    def forward(self, x):

        x = self.gconv1(x)
        x = g.GConv2d_MaxPooling(x, 2)

        x = torch.max(x, dim=2)[0]
        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        x = torch.sort(x, dim=2)[0]
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

class P4_lastlayer(nn.Module):

    def __init__(self, image_size, in_channels, out_channels, output_features, kernel_size):

        super(P4_lastlayer, self).__init__()
        self.gconv1 = g.P4P4GConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.fc = nn.Linear(out_channels * int(image_size/2) * int(image_size/2), output_features)
    
    def forward(self, x):

        x = self.gconv1(x)
        x = g.GConv2d_MaxPooling(x, 2)

        x = torch.max(x, dim=2)[0]
        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))
        x = torch.sort(x, dim=2)[0]
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


in_channels = 10
out_channels = 10
output_features = 10
kernel_size = 3
batch_size = 10
image_size = 28
Z2tolerance = 1.0e-4
P4tolerance = 1.0e-5

check_invariance('Z2', in_channels, out_channels, output_features, kernel_size, batch_size, image_size, Z2tolerance=Z2tolerance, P4tolerance=P4tolerance)
print('True for Z2')
check_invariance('P4', in_channels, out_channels, output_features, kernel_size, batch_size, image_size, Z2tolerance=Z2tolerance, P4tolerance=P4tolerance)
print('True for P4')
