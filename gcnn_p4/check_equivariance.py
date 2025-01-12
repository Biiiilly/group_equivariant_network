import pytest
import torch
from gcnn_p4.layer_p4 import Z2P4GConv2d, P4P4GConv2d

batch_size = 64
image_size = 28
kernel_size = 3

@pytest.mark.parametrize('in_channels, out_channels', [(10, 20), (20, 30), (30, 40)])
def test_Z2P4equivariant(in_channels, out_channels):
    x = torch.randn(batch_size, in_channels, image_size, image_size)
    network = Z2P4GConv2d(in_channels, out_channels, kernel_size)
    result = network(x)
    for i in range(1, 4):
        result_rotate = torch.rot90(result, i, (3, 4))
        x_rotate = torch.rot90(x, i, (2, 3))
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
                for _ in range(4):
                    if torch.allclose(sorted1[i], sorted2[i], atol=1e-5) == False:
                        difference = torch.abs(sorted1[i] - sorted2[i])
                        print("Max absolute difference:", torch.max(difference))
                    assert torch.allclose(sorted1[i], sorted2[i], atol=1e-4)

@pytest.mark.parametrize('in_channels, out_channels', [(10, 20), (20, 30), (30, 40)])
def test_P4P4equivariant(in_channels, out_channels):
    x = torch.randn(batch_size, in_channels, 4, image_size, image_size)
    network = P4P4GConv2d(in_channels, out_channels, kernel_size)
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
                for _ in range(4):
                    if torch.allclose(sorted1[i], sorted2[i], atol=1e-4) == False:
                        difference = torch.abs(sorted1[i] - sorted2[i])
                        print("Max absolute difference:", torch.max(difference))
                    assert torch.allclose(sorted1[i], sorted2[i], atol=1e-4)

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
