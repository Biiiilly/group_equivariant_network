import torch
import torch.nn as nn
import torch.nn.functional as F
import P4GConv2d as g


# Check Equivariance for the second layer
def P4P4_check_equivariant(in_channels, out_channels, image_size, batch_size, kernel_size):

    x = torch.randn(batch_size, in_channels, 4, image_size, image_size)
    network = g.P4P4GConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    result = network(x)

    for i in range(1, 4):

        result_rotate = torch.rot90(result, i, (3, 4))
        x_rotate = torch.rot90(x, i, (3, 4))
        rotate_result = network(x_rotate)

        for b in range(batch_size):
            for c in range(out_channels):

                result_rotate_fixed = result_rotate[b][c].reshape(4, -1)
                rotate_result_fixed = rotate_result[b][c].reshape(4, -1)

                first_column1 = result_rotate_fixed[:, 0]
                first_column2 = rotate_result_fixed[:, 0]

                sorted_indices1 = torch.argsort(first_column1)
                sorted1 = result_rotate_fixed[sorted_indices1]
                sorted_indices2 = torch.argsort(first_column2)
                sorted2 = rotate_result_fixed[sorted_indices2]

                for ind in range(4):
                    if torch.allclose(sorted1[i], sorted2[i], atol=1e-4) == False:
                        print(f'b:{b}, c:{c}, i:{i}')
                        print(sorted1[i])
                        print(sorted2[i])
                        difference = torch.abs(sorted1[i] - sorted2[i])
                        print("Max absolute difference:", torch.max(difference))
                    assert torch.allclose(sorted1[i], sorted2[i], atol=1e-4)


in_channels = 10
out_channels = 12
image_size = 4
kernel_size = 3
batch_size = 10

P4P4_check_equivariant(in_channels, out_channels, image_size, batch_size, kernel_size)
print('Equivariant for P4P4 Layer')