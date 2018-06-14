from unittest import TestCase
from typing import Union, Sequence

import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

from torch_same_pad import get_pad
from torch_same_pad.pad import _get_compressed


class TestPad2D(TestCase):

    def _check(self,
               height: int,
               width: int,
               channels: int,
               kernel_size: Union[int, Sequence[int]] = 3,
               stride: Union[int, Sequence[int]] = 1,
               dilation: Union[int, Sequence[int]] = 1,
               batch_size: int = 2,
               filters: int = 5):
        args = f'height={height}, width={width}, channels={channels}, '\
            f'kernel_size={kernel_size}, stride={stride}, '\
            f'dilation={dilation}, filters={filters}'
        inputs = np.random.standard_normal((batch_size, height, width, channels))

        weight = np.random.standard_normal((_get_compressed(kernel_size, 0),
                                            _get_compressed(kernel_size, 1),
                                            channels,
                                            filters))
        bias = np.random.standard_normal((filters,))

        keras_input = tf.keras.layers.Input(shape=(height, width, channels))
        keras_conv = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=stride,
                                            dilation_rate=dilation,
                                            padding='same')
        keras_model = tf.keras.models.Model(keras_input, keras_conv(keras_input))
        keras_conv.set_weights([weight, bias])
        keras_output = keras_model.predict(inputs)

        torch_input = torch.from_numpy(inputs.transpose((0, 3, 1, 2))).to(torch.float32)
        torch_pad = get_pad(size=(height, width),
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=dilation)
        args += f' pad={torch_pad}'
        torch_conv = torch.nn.Conv2d(in_channels=channels,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     dilation=dilation)
        torch_conv.state_dict()['weight'][:] = torch.from_numpy(weight.transpose((3, 2, 0, 1)))
        torch_conv.state_dict()['bias'][:] = torch.from_numpy(bias)
        torch_padded = F.pad(torch_input, pad=torch_pad)
        torch_output = torch_conv(torch_padded).detach().numpy().transpose((0, 2, 3, 1))

        self.assertEqual(keras_output.shape, torch_output.shape, args + f' padded={torch_padded.shape}')

        diff = np.abs(keras_output - torch_output).max()
        self.assertLess(diff, 1e-4, args)

    def test_base_1(self):
        self._check(height=4, width=5, channels=6)

    def test_base_2(self):
        self._check(height=4, width=4, channels=4, kernel_size=(4, 5), filters=3)

    def test_enum(self):
        for length in range(3, 7):
            for channels in range(3, 5):
                for kernel_size in range(3, 6):
                    for stride in range(1, 4):
                        dilation_range = range(1, 4)
                        if stride > 1:
                            dilation_range = [1]
                        for dilation in dilation_range:
                            for filters in range(2, 4):
                                self._check(height=length,
                                            width=length,
                                            channels=channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilation=dilation,
                                            filters=filters)

    def test_random(self):
        for _ in range(100):
            height = np.random.randint(3, 7)
            width = np.random.randint(3, 7)
            channels = np.random.randint(3, 5)
            kernel_size = (np.random.randint(3, 6), np.random.randint(3, 6))
            if np.random.randint(0, 2) == 0:
                stride = (np.random.randint(1, 4), np.random.randint(1, 4))
                dilation = 1
            else:
                stride = 1
                dilation = (np.random.randint(1, 4), np.random.randint(1, 4))
            filters = np.random.randint(2, 4)
            self._check(height=height,
                        width=width,
                        channels=channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        filters=filters)
