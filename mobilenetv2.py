import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def _conv1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class _InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(_InvertedResidual, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                # pointwise
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            hidden_dim = round(in_channels * expand_ratio)
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, bias=False, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        first_channels = 32
        inverted_residual_config = [
            # t (expand ratio), channels, n (layers), stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = [_conv_bn(3, first_channels, stride=2)]
        input_channels = first_channels
        for i, (t, c, n, s) in enumerate(inverted_residual_config):
            output_channels = c
            layers = []
            for j in range(n):
                if j == 0:
                    layers.append(_InvertedResidual(input_channels, output_channels, stride=s, expand_ratio=t))
                else:
                    layers.append(_InvertedResidual(input_channels, output_channels, stride=1, expand_ratio=t))
                input_channels = output_channels
            features.append(nn.Sequential(*layers))
        last_channels = 1280
        features.append(_conv1x1_bn(input_channels, last_channels))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(last_channels, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out
