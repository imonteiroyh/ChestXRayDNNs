import unittest

import torch
import torch.nn as nn

from xrkit.models.densenet import DenseNet201
from xrkit.models.inceptionv3 import InceptionV3
from xrkit.models.nasnet import NASNetLarge
from xrkit.models.resnet import ResNet152V2
from xrkit.models.unet import UNet
from xrkit.models.vgg19 import VGG19
from xrkit.models.xception import Xception
from xrkit.segmentation import (
    average_surface_distance,
    balanced_average_hausdorff_distance,
    dice,
    jaccard_index,
)

task_models = [DenseNet201, InceptionV3, NASNetLarge, ResNet152V2, VGG19, Xception]
segmentation_models = [UNet]
metrics = [average_surface_distance, balanced_average_hausdorff_distance, dice, jaccard_index]


class SegmentationModelTest(unittest.TestCase):
    def test_basic(self):
        input_shape = (4, 3, 128, 128)
        input_tensor = torch.randn(input_shape)

        for n_outputs in (1, 3):

            for model in task_models + segmentation_models:

                if model in segmentation_models:
                    current_model = model(n_outputs=n_outputs)
                else:
                    current_model = model(task="segmentation", n_outputs=n_outputs)

                with self.subTest(model=current_model):
                    output = current_model(input_tensor)
                    self.assertEqual(output.shape, (4, n_outputs, 128, 128))

    def test_metrics(self):
        for metric in metrics:
            for n_outputs in [1, 2, 3, 5, 10]:
                input_shape = (4, n_outputs, 128, 128)

                mask = (torch.randn(input_shape) > 0.5).int()
                predictions = torch.randn(input_shape)

                try:
                    metric(predictions, mask)
                except:
                    self.fail(f"Error calculating {metric.__name__} for n_outputs={n_outputs}")
