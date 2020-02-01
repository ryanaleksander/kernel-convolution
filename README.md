# Introduction
PyTorch implementation of Kervolutional Neural Network (CVPR 2019): https://arxiv.org/abs/1904.03955

# Requirements
The code is written using the following environment. There isn't a strict version requirement, but deviate from the listed versions at your own risk

* python: 3.7.3
* pytorch: 1.2.0
* torchvision: 0.4.0

# Tutorials
## Creating a kernel convolution layer
```python
from kernels import *
from kernel_conv import KernelConv2d

# Create a kernel to use for your convolution layers
# There are currently 3 to choose from
gaussian_kernel = GaussianKernel(gamma=0.5)
polynomial_kernel = PolynomialKernel(c=0, degree=3)
sigmoid_kernel = SigmoidKernel()

# Create a KernelConv2d like you would a Conv2d, and pass
# in the kernel
kernel_conv2d = KernelConv2d(3, 64, (3,3), kernel=gaussian_kernel)
```
## Converting existing networks
```python
from kernels import *
from kernel_conv import kernel_wrapper

# Let's use ResNet50 as an example
import torchvision
resnet50 = torchvision.models.resnet50()

# Create a kernel to replace all the existing Conv2d layers
kernel = GaussianKernel()

# Apply wrapper function
kernel_wrapper(resnet50)
```