import torch
import torch.nn.functional as F

def GConv2d_MaxPooling(x, kernel_size):

    size = x.size()
    x = x.view(size[0], size[1]*size[2], size[3], size[4])
    x = F.max_pool2d(x, kernel_size)
    x = x.view(size[0], size[1], size[2], int(size[3]/kernel_size), int(size[4]/kernel_size))
    x = torch.max(x, dim=2)[0]

    return x
