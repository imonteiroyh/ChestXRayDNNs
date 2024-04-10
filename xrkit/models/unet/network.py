import torch
import torch.nn as nn

from xrkit.models.unet.block import ConvBlock, DecoderBlock, EncoderBlock


class UNet(nn.Module):
    def __init__(self, n_inputs: int = 1) -> None:
        super().__init__()

        self.encoder1 = EncoderBlock(n_inputs, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        self.outputs = nn.Conv2d(64, n_inputs, kernel_size=1, padding=0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        skip1, tensor = self.encoder1(tensor)
        skip2, tensor = self.encoder2(tensor)
        skip3, tensor = self.encoder3(tensor)
        skip4, tensor = self.encoder4(tensor)

        tensor = self.bottleneck(tensor)

        tensor = self.decoder1(tensor, skip4)
        tensor = self.decoder2(tensor, skip3)
        tensor = self.decoder3(tensor, skip2)
        tensor = self.decoder4(tensor, skip1)

        outputs = self.outputs(tensor)

        return outputs
