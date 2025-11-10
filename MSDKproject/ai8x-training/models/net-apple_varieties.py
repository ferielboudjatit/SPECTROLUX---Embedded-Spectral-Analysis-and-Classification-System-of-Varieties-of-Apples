###################################################################################################
#
# Copyright (C) 2025 Your Name / Institution. All Rights Reserved.
#
###################################################################################################
"""
Apple Variety Discrimination using 1D CNN (AI8X Optimized)
"""
import torch
import torch.nn as nn
import ai8x


class AI85AppleNet(nn.Module):
    """
    AI85 Optimized 1D CNN for Apple Variety Classification
    """

    def __init__(self, num_classes=4, num_channels=1, dimensions = (128,1), input_length=288, bias=False, **kwargs):
        super().__init__()

        # First 1D Convolution Layer
        self.conv1 = ai8x.FusedConv1dReLU(num_channels, 32, kernel_size=5, stride=1, padding=2,
                                          bias=bias, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv1dReLU(32, 64, kernel_size=3, stride=1, padding=1,
                                        bias=bias, **kwargs)
            
        # Second 1D Convolution Layer
        self.conv3 = ai8x.FusedMaxPoolConv1dReLU(64, 14, kernel_size=3, stride=1, padding=1,
                                          bias=bias, **kwargs)

        # Fully Connected Layers
        self.fc1 = ai8x.Linear(1008, 32, bias=bias, wide=True, **kwargs)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = ai8x.Linear(32, num_classes, bias=bias, wide=True, **kwargs)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """Forward pass"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def ai85apple_discrimination(pretrained=False, **kwargs):
    """
    Constructs the Apple Discrimination model.
    """
    assert not pretrained
    return AI85AppleNet(**kwargs)


# Register model in AI8X training
models = [
    {
        'name': 'ai85apple_discrimination',
        'min_input': 1,  # 1D data
        'dim': 1,        # 1D Conv model
    },
]

