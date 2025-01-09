import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcnn_p4.network import P4GConvNet

def test_invariance():
    network = P4GConvNet()
    x = torch.randn(1, 1, 28, 28)
    y = network(x)
    for i in range(3):
        x_rot = torch.rot90(x, i, (2, 3))
        y_rot = network(x_rot)
        assert torch.allclose(y, y_rot, atol=1e-5)

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
