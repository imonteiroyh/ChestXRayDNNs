import unittest

import torch
import torch.nn as nn

from xrkit.models.densenet import DenseNet201
from xrkit.models.nasnetlarge import NASNetLarge
from xrkit.models.resnet152v2 import ResNet152V2
from xrkit.models.unet import UNet
from xrkit.models.vgg19 import VGG19

models = [DenseNet201, NASNetLarge, ResNet152V2, UNet, VGG19]


class ModelTest(unittest.TestCase):
    def test_basic(self):
        for n_channels in (1, 3):
            input_shape = (4, n_channels, 128, 128)
            input_tensor = torch.randn(input_shape)

            for model in models:
                current_model = model(n_inputs=n_channels)
                with self.subTest(model=current_model):
                    output = current_model(input_tensor)
                    self.assertEqual(output.shape, input_shape)
