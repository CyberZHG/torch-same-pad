from typing import Union, Sequence

__all__ = ['get_pad_1d']


def _calc_pad(size: int,
              kernel_size: int = 3,
              stride: int = 1,
              dilation: int = 1):
    pad = (((size + stride - 1) // stride - 1) * stride + kernel_size - size) * dilation
    return pad // 2, pad - pad // 2


def get_pad_1d(size: int,
               kernel_size: int = 3,
               stride: int = 1,
               dilation: int = 1):
    return _calc_pad(size, kernel_size, stride, dilation)
