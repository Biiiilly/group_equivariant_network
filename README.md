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
│   ├── check_equivariance.py        # Tests equivariance properties of the layers
│   ├── check_invariance.py          # Tests invariance properties of the network
│   ├── layer_p4.py                  # Implements P4 group convolution layers
│   ├── max_pool_p4.py               # Implements group-based max pooling
│   ├── network.py                   # Defines G-CNN and standard CNN architectures
├── training/
│   ├── gcnn_mnist_test.py           # G-CNN Training and evaluation on standard MNIST
│   ├── cnn_mnist_test.py            # CNN Training and evaluation on standard MNIST
│   ├── gcnn_rot_mnist_test.py       # G-CNN Training and evaluation on rotated MNIST
│   ├── test.py                      # Comparison of G-CNN and CNN
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

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Biiiilly/group_equivariant_network
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
python training/gcnn_mnist_test.py
```

### Training on Rotated MNIST
Run the training script for rotated MNIST:
```bash
python training/gcnn_rot_mnist_test.py
```

### Testing Invariance
Check rotational invariance of the model:
```bash
python -m pytest gcnn_p4/check_invariance.py
```

### Testing Equivariance
Check rotational Equivariance of the layers:
```bash
python -m pytest gcnn_p4/check_equivariance.py
```

## Results
### G-CNN Trained on normal MNIST dataset
- **Structure**: Two intermediate layers with 20 filters for each.
- **Accuarcy**: Around 86%.
- **Evaluation on rotated MNIST dataset**: Around 57.98% compared with 53.97% tested on CNN.
- **Evaluation on 90 degrees rotated MNIST dataset**: Around 86.34% compared with 18.26% tested on CNN.

## Conclusion
This project demonstrates the effectiveness of Group Equivariant Convolutional Networks (G-CNNs) in leveraging rotational symmetries for improved image classification. G-CNNs outperform standard CNNs, particularly on rotated datasets, showcasing their robustness and efficiency in symmetry-aware tasks. With a modular design and rigorous testing, this implementation highlights the potential of equivariant methods for advancing deep learning.
