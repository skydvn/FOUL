
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_bn=True,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        if has_bn:
            self.bn2 = norm_layer(planes)
        else:
            self.bn2 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if has_bn:
            self.bn3 = norm_layer(planes)
        else:
            self.bn3 = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_bn=True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if has_bn:
            self.bn1 = norm_layer(width)
        else:
            self.bn1 = nn.Identity()
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if has_bn:
            self.bn2 = norm_layer(width)
        else:
            self.bn2 = nn.Identity()
        self.conv3 = conv1x1(width, planes * self.expansion)
        if has_bn:
            self.bn3 = norm_layer(planes * self.expansion)
        else:
            self.bn3 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


def forward(self, x: Tensor) -> Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class UResNet(nn.Module):

    def __init__(
            self,
            block: BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_bn=True,
            bn_block_num=4,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if has_bn:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []
        self.layers.extend(self._make_layer(block, 64, layers[0], has_bn=has_bn and (bn_block_num > 0)))
        for num in range(1, len(layers)):
            self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
                                                dilate=replace_stride_with_dilation[num - 1],
                                                has_bn=has_bn and (num < bn_block_num)))

        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(features[len(layers) - 1] * block.expansion, num_classes)

        # self.fc = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(features[len(layers)-1] * block.expansion, num_classes)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        self.inv_encoder = nn.Sequential() ## creating the invariant encoder
        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            self.inv_encoder.add_module(f'inv_layer_{i}', layer)

        self.var_encoder = nn.Sequential() ## varaint encoder
        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            self.var_encoder.add_module(f'var_layer_{i}', layer)

        #decoder reconstruction head
        self.decoder = nn.Sequential(
        # Upsample and reduce channels gradually
        nn.ConvTranspose2d(features[len(layers)-1] * block.expansion, features[2], kernel_size=2, stride=2),
        nn.BatchNorm2d(features[2]),
        nn.ReLU(inplace=True),
        
        nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2),
        nn.BatchNorm2d(features[1]),
        nn.ReLU(inplace=True),
        
        nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2),
        nn.BatchNorm2d(features[0]),
        nn.ReLU(inplace=True),
        
        # Final reconstruction to original image size and channels
        nn.ConvTranspose2d(features[0], 3, kernel_size=2, stride=2),
        nn.Tanh()  # or nn.Sigmoid() depending on your input normalization
    )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(features[len(layers)-1] * block.expansion, num_classes)
        )

    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, has_bn=True) -> List:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_bn:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.Identity(),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, has_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, has_bn=has_bn))

        return layers

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = x
        x2 = x

        """
        Wrap into Invariant Encoder
        """
        x1 = self.inv_encoder(x1)

        """
        Wrap into Variant Encoder
        """
        x2 = self.var_encoder(x2)

        """
        Add a reconstruction decoder head
        """
        combined_features = torch.cat([x1, x2], dim=1) ## concat the invariant and variant features 
        x_rec = self.decoder(combined_features)

        """
        Wrap into Classifier
        """
        x = self.classifier(x)

        return x, x_rec

    def _forward_impl_exp(self, x: Tensor) -> Tensor:
        x = self.general_enc(x)
        x_inv = self.inv_enc(x)
        x_var = self.var_enc(x)
        x_tot = torch.concat(x_inv, x_var) # Use torch.concat
        y = self.classifier(x_tot)
        x_rec = self.reconstructor(x_tot)

        return y, x_rec

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def uresnet152(**kwargs: Any) -> ResNet:
    return UResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def uresnet101(**kwargs: Any) -> ResNet:
    return UResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def uresnet50(**kwargs: Any) -> ResNet:
    return UResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def uresnet34(**kwargs: Any) -> ResNet:
    return UResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def uresnet18(**kwargs: Any) -> ResNet:  # 18 = 2 + 2 * (2 + 2 + 2 + 2)
    return UResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def uresnet10(**kwargs: Any) -> ResNet:  # 10 = 2 + 2 * (1 + 1 + 1 + 1)
    return UResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def uresnet8(**kwargs: Any) -> ResNet:  # 8 = 2 + 2 * (1 + 1 + 1)
    return UResNet(BasicBlock, [1, 1, 1], **kwargs)


def uresnet6(**kwargs: Any) -> ResNet:  # 6 = 2 + 2 * (1 + 1)
    return UResNet(BasicBlock, [1, 1], **kwargs)


def uresnet4(**kwargs: Any) -> ResNet:  # 4 = 2 + 2 * (1)
    return UResNet(BasicBlock, [1], **kwargs)
