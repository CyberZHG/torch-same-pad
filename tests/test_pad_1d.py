from unittest import TestCase

import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

from torch_same_pad import get_pad


class TestPad1D(TestCase):

    def _check(self,
               length: int,
               channels: int,
               kernel_size: int = 3,
               stride: int = 1,
               dilation: int = 1,
               batch_size: int = 2,
               filters: int = 5):
        args = f'length={length}, channels={channels}, '\
            f'kernel_size={kernel_size}, stride={stride}, '\
            f'dilation={dilation}, filters={filters}'
        inputs = np.random.standard_normal((batch_size, length, channels))

        weight = np.random.standard_normal((kernel_size, channels, filters))
        bias = np.random.standard_normal((filters,))

        keras_input = tf.keras.layers.Input(shape=(length, channels))
        keras_conv = tf.keras.layers.Conv1D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=stride,
                                            dilation_rate=dilation,
                                            padding='same')
        keras_model = tf.keras.models.Model(keras_input, keras_conv(keras_input))
        keras_conv.set_weights([weight, bias])
        keras_output = keras_model.predict(inputs)

        torch_input = torch.from_numpy(inputs.transpose((0, 2, 1))).to(torch.float32)
        torch_pad = get_pad(size=length,
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=dilation)
        args += f' pad={torch_pad}'
        torch_conv = torch.nn.Conv1d(in_channels=channels,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     dilation=dilation)
        torch_conv.state_dict()['weight'][:] = torch.from_numpy(weight.transpose((2, 1, 0)))
        torch_conv.state_dict()['bias'][:] = torch.from_numpy(bias)
        torch_output = F.pad(torch_input, pad=torch_pad)
        torch_output = torch_conv(torch_output).detach().numpy().transpose((0, 2, 1))

        self.assertEqual(keras_output.shape, torch_output.shape, args)

        diff = np.abs(keras_output - torch_output).max()
        self.assertLess(diff, 1e-4, args)

    def test_base(self):
        self._check(length=5, channels=6)

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
                                self._check(length=length,
                                            channels=channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilation=dilation,
                                            filters=filters)
