import torch
import torch.nn as nn

from resnet import _resnet, Bottleneck


def attention(in_planes, num_heads):
    return nn.MultiheadAttention(in_planes, num_heads, dropout=0.0)


class AttBottleneck(Bottleneck):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_heads=8):
        super(AttBottleneck, self).__init__(inplanes, planes, stride=stride,
                downsample=downsample, groups=groups, base_width=base_width,
                dilation=dilation, norm_layer=None)
        # Replace conv2 in the bottleneck with attention.
        width = int(planes * (base_width / 64.)) * groups
        self.att2 = attention(width, num_heads) 
        if stride == 1:
            self.avgpool_after_att = nn.Identity()
        else:
            self.avgpool_after_att = nn.AvgPool1d(kernel_size=3, stride=stride,
                    padding=dilation)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = out.permute(2, 0, 1)
        out, _ = self.att2(out, out, out)
        out = out.permute(1, 2, 0)
        out = self.avgpool_after_att(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def resnatt18(**kwargs):
    r"""ResNet-18 model architecture adapted from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet18', AttBottleneck, [2, 2, 2, 2], **kwargs)


def resnatt50(**kwargs):
    r"""ResNet-50 model architecture adapted from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet50', AttBottleneck, [3, 4, 6, 3], **kwargs)


RESNATT_NAME_TO_MODEL_MAP = {
        'resnatt18': resnatt18,
        'resnatt50': resnatt50,
}
