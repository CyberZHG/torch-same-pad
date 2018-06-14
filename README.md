# Torch SAME Padding

![License](https://img.shields.io/pypi/l/keras-bert.svg)

The padding calculation used for converting TensorFlow convolution or pooling layers with `SAME` padding to PyTorch.

## Install

```bash
pip install git+https://github.com/CyberZHG/torch-same-pad.git
```

## Usage

```python
import torch
import torch.nn.functional as F

from torch_same_pad import get_pad, pad


x = torch.Tensor()


# Use `get_pad` to calculate the padding
torch_pad = get_pad(size=20,
                    kernel_size=3,
                    stride=1,
                    dilation=1)
torch_output = F.pad(x, pad=torch_pad)


# Use `pad` to do the padding directly
torch_padded = pad(x,
                   size=(224, 224),
                   kernel_size=3,
                   stride=1,
                   dilation=1)
```
