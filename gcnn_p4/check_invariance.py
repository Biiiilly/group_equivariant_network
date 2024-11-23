import torch
import torch.nn as nn
import torch.nn.functional as F
import gcnn_p4.P4GConv2d as g

def check_invariant_1(network, in_channel, out_channel, image_size, batch_size):

    x = torch.randn(batch_size, in_channel, image_size, image_size)
    result = network(x) # [batch_size, out_channel, image_size, image_size]

    for i in range(1, 4):

        # result_rotate = torch.rot90(result, i, (2, 3))
        x_rotate = torch.rot90(x, i, (2, 3))
        rotate_result = network(x_rotate) # [batch_size, out_channel, image_size, image_size]

        for b in range(batch_size):
            for c in range(out_channel):

                a1 = result[b][c]
                a2 = rotate_result[b][c]

                if torch.allclose(a1, a2) == False:
                    print(a1)
                    print(a2)
                    print(f'b:{b}, c:{c}')

                assert torch.allclose(a1, a2)

'''
def check_invariant_2(network, in_channel, out_channels, image_size, batch_size):

    x = torch.randn(batch_size, in_channel, image_size, image_size)
    result = network(x)

    x1 = result.view(result.size(0), -1)
    fc = nn.Linear(out_channels * image_size * image_size, 10)
    x1 = fc(x1)

    for i in range(1, 4):
        
        x_rotate = torch.rot90(x, i, (2, 3))
        rotate_result = network(x_rotate)

        if torch.allclose(result, rotate_result, atol=1e-5) == False:
            print(result)
            print(rotate_result)
            print(torch.norm(result-rotate_result)) 

        assert torch.allclose(result, rotate_result)

        x2 = rotate_result.view(x.size(0), -1)
        x2 = fc(x2)

        if torch.allclose(x1, x2, atol=1e-5) == False:
            print(x1)
            print(x2)
            print(torch.norm(x1-x2))

        assert torch.allclose(x1, x2, atol=1e-5)
'''

class test_P4GConvNet_lastlayer(nn.Module):

    def __init__(self, image_size=6, in_channels=1, out_channels=3, output_features=5, kernel_size=3):

        super(test_P4GConvNet_lastlayer, self).__init__()
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


model = test_P4GConvNet_lastlayer(image_size=6, in_channels=1, out_channels=3, output_features=5, kernel_size=3)
# check_invariant_1(model, in_channel=1, out_channel=3, image_size=5, batch_size=1)
# print('True for model')
