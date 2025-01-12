# Group Equivariant Convolutional Networks (G-CNN)

This repository implements a Group Equivariant Convolutional Network (G-CNN) for image classification tasks, inspired by the paper "Group Equivariant Convolutional Networks" by Taco S. Cohen and Max Welling ([arXiv:1602.07576v3](https://arxiv.org/abs/1602.07576v3)). This implementation focuses on leveraging rotational symmetries to improve performance on datasets such as MNIST and rotated MNIST.
This is part of my M4R project focusing on symmetries in deep learning. I conducted this research from October 2024 at Imperial College London, under the supervision of Dr. Webster Kevin and Professor Jeroen Lamb.

## Overview
G-CNNs extend the traditional convolutional neural network (CNN) paradigm by incorporating group equivariant convolutions. These layers exploit symmetries, such as translations and rotations, enabling the network to share weights more effectively and generalize better with fewer parameters. This repository includes the following components:

- **Custom convolutional layers**: Implements group convolution layers for rotational symmetries (p4 group).
- **Network architectures**: Includes both G-CNN and standard CNN models for comparison.
- **Datasets**: Training and evaluation on MNIST and rotated MNIST datasets.
- **Performance metrics**: Tracks accuracy and loss over epochs for visualization and comparison.

## Project Structure

```
├── gcnn_p4/
│   ├── __init__.py                  # Initializes the package
│   ├── check_equivariance.py        # Tests equivariance properties of the G-CNN
│   ├── check_invariance.py          # Tests invariance properties of the G-CNN
│   ├── layer_p4.py                  # Implements P4 group convolution layers
│   ├── max_pool_p4.py               # Implements group-based max pooling
│   ├── network.py                   # Defines G-CNN and standard CNN architectures
├── training/
│   ├── mnist_test.py                # Training and evaluation on standard MNIST
│   ├── mnist_test_rot.py            # Evaluation on rotated MNIST
│   ├── rot_mnist_test.py            # Training and evaluation on rotated MNIST
├── reference/                       # Contains reference research papers
│   ├── 1602.07576v3.pdf
│   ├── 2106.06610v4.pdf
│   ├── 2209.14991v3.pdf
├── requirements.txt                 # Lists the required dependencies
├── setup.py                         # Script for setting up the project
└── README.md                        # Documentation of the project
```

## Key Features

### 1. Group Convolution Layers
The custom layers in `layer_p4.py` include:
- **Z2P4GConv2d**: Transforms from standard convolution (Z2) to group convolution (p4).
- **P4P4GConv2d**: Operates entirely in the p4 group space, applying convolutions that are equivariant to rotations.

### 2. Group Max Pooling
The pooling operation in `max_pool_p4.py` ensures equivariance by pooling over the group dimension.

### 3. Network Architectures
In `network.py`, two main architectures are defined:
- **P4GConvNet**: A G-CNN model utilizing p4 group convolutions.
- **CNN**: A standard convolutional neural network for baseline comparison.

### 4. MNIST and Rotated MNIST
Scripts for training and testing the models include:
- `mnist_test.py`: Training and evaluating models on the standard MNIST dataset.
- `mnist_test_rot.py`: Testing pre-trained G-CNNs on rotated MNIST.
- `rot_mnist_test.py`: Training and evaluating models on rotated MNIST.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Biiiilly/group_equivariant_network
   cd gcnn
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have PyTorch installed with GPU support if available.

## Usage

### Training on MNIST
Run the training script for standard MNIST:
```bash
python mnist_test.py
```

### Training on Rotated MNIST
Run the training script for rotated MNIST:
```bash
python rot_mnist_test.py
```

### Testing Invariance
Check rotational invariance of the model:
```bash
python check_invariance.py
```

## Results
### MNIST
- **Standard CNN**: Baseline accuracy on MNIST.
- **P4GConvNet**: Improved accuracy by leveraging group convolutions.

### Rotated MNIST
- **Standard CNN**: Limited performance due to lack of rotational equivariance.
- **P4GConvNet**: Achieves significantly better accuracy due to rotational equivariance.

## Reference
If you use this code, please cite:
```
@article{cohen2016group,
  title={Group Equivariant Convolutional Networks},
  author={Cohen, Taco S and Welling, Max},
  journal={arXiv preprint arXiv:1602.07576},
  year={2016}
}
```

## Acknowledgments
This implementation is based on the foundational work of Cohen and Welling on Group Equivariant Convolutional Networks. Special thanks to the PyTorch community for providing excellent tools for neural network development.

