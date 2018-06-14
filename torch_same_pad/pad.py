import collections
from typing import Union, Sequence

__all__ = ['get_pad']


def _calc_pad(size: int,
              kernel_size: int = 3,
              stride: int = 1,
              dilation: int = 1):
    pad = (((size + stride - 1) // stride - 1) * stride + kernel_size - size) * dilation
    return pad // 2, pad - pad // 2


def _get_compressed(item: Union[int, Sequence[int]], index: int):
    if isinstance(item, collections.Sequence):
        return item[index]
    return item


def get_pad(size: Union[int, Sequence[int]],
            kernel_size: Union[int, Sequence[int]] = 3,
            stride: Union[int, Sequence[int]] = 1,
            dilation: Union[int, Sequence[int]] = 1):
    len_size = 1
    if isinstance(size, collections.Sequence):
        len_size = len(size)
    pads = []
    for i in range(len_size):
        pads.append(_calc_pad(size=_get_compressed(size, i),
                              kernel_size=_get_compressed(kernel_size, i),
                              stride=_get_compressed(stride, i),
                              dilation=_get_compressed(dilation, i)))
    if len_size == 1:
        pad = pads[0]
    elif len_size == 2:
        pad = pads[1] + pads[0]
    elif len_size == 3:
        pad = pads[1] + pads[0] + pads[2]
    else:
        raise NotImplementedError
    return pad
