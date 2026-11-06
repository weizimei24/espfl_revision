import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


# VGG-D (VGG16) 标准通道配置
VGG_CFGS = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
}


def _make_layers(
    cfg: List[Union[int, str]],
    batch_norm: bool = True,
    track_running_stats: bool = False,
    slim_idx: int = 0,
    scale: float = 1.0,
) -> nn.Sequential:
    """
    构建 VGG features 部分。

    参数：
        cfg                 : 通道配置列表，'M' 表示 MaxPool
        batch_norm          : 是否加 BN
        track_running_stats : BN 是否维护 running_mean/var（联邦学习统一 False）
        slim_idx            : 从第几个 Conv 层开始应用 scale 缩放（0=全部缩放）
        scale               : 宽度缩放比例（0.5 / 0.71 / 1.0）
    """
    layers = []
    in_channels = 3
    conv_idx = 0

    cfg = list(cfg) + ['M']

    for x in cfg:
        if x == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        out_channels = int(x * scale) if conv_idx >= slim_idx else x
        out_channels = max(1, out_channels)   # 防止 scale 极小时通道数为 0

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(
                out_channels,
                track_running_stats=track_running_stats,
            ))
        layers.append(nn.ReLU(inplace=True))

        in_channels = out_channels
        conv_idx += 1

    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """
    宽度可缩放的 VGG16。

    参数：
        num_classes         : 分类头输出维度
        scale               : 宽度缩放比例（1.0=标准, 0.71=中, 0.5=小）
        slim_idx            : 从第几个 Conv 层开始应用 scale（默认 0=全部）
        track_running_stats : BN running stats 开关（联邦学习统一 False）

    参数量（CIFAR-10, scale=1.0）：
        features  ≈ 14.7M
        projector ≈ 18.9M  (512×4 → 4096 → 4096)
        classifier≈ 0.04M
        总计      ≈ 33.6M
    """

    def __init__(
        self,
        num_classes: int = 10,
        scale: float = 1.0,
        slim_idx: int = 0,
        track_running_stats: bool = False,
    ):
        super(VGG16, self).__init__()
        self.scale = scale

        self.features = _make_layers(
            VGG_CFGS['D'],
            batch_norm=True,
            track_running_stats=track_running_stats,
            slim_idx=slim_idx,
            scale=scale,
        )

        # 动态推导展平维度（适配任意 scale 和输入尺寸）
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feat_dim = self.features(dummy).view(1, -1).size(1)

        # scale=1.0 时 feat_dim=2048，dim=4096，与标准 VGG16 classifier 对齐
        dim = int(4096 * scale)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return self.classifier(x)

    def get_conv_layer_names(self):
        return [
            name for name, m in self.named_modules()
            if isinstance(m, nn.Conv2d)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# ResNet18（CIFAR 风格，完全自实现）
# ─────────────────────────────────────────────────────────────────────────────

class _ResidualBlock(nn.Module):
    """
    标准 BasicBlock（CIFAR 风格），支持通道数/stride 不对齐时的 shortcut 投影。
    """

    def __init__(
        self,
        inchannel: int,
        outchannel: int,
        stride: int = 1,
        track_running_stats: bool = False,
    ):
        super().__init__()
        trs = track_running_stats
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=trs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=trs),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=trs),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.left(x)
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetFed(nn.Module):
    """
    联邦学习用 CIFAR 风格 ResNet18（完全自实现，不依赖 torchvision）。

    设计原则：
        - 纯自实现 BasicBlock，完全掌控 BN 的 track_running_stats
        - 支持宽度缩放（scale）和分层缩放起始索引（slim_idx）
        - 适配小尺寸输入（CIFAR 32×32 / FEMNIST 28×28）：
          stem 使用 3×3 Conv stride=1，无 MaxPool

    层索引（slim_idx 对应）：
        0 = stem (conv1,  64 通道基准)
        1 = layer1 (96   通道, 2 blocks, stride=1)
        2 = layer2 (128  通道, 2 blocks, stride=2)
        3 = layer3 (256  通道, 2 blocks, stride=2)
        4 = layer4 (512  通道, 2 blocks, stride=2)
        5 = fc

    参数：
        num_classes         : 分类头输出维度
        scale               : 宽度缩放比例（1.0 / 0.71 / 0.5 …）
        slim_idx            : 从第几层开始应用 scale（0=全部缩放）
        track_running_stats : BN running stats 开关（联邦学习统一 False）
        in_channels         : 输入通道数（RGB=3, 灰度=1）
        depth / input_size  : 兼容旧调用接口保留，不影响网络结构
    """

    _BASE_CHANNELS = [64, 96, 128, 256, 512]   # idx: stem, layer1~4

    def __init__(
        self,
        num_classes: int = 10,
        scale: float = 1.0,
        slim_idx: int = 0,
        track_running_stats: bool = False,
        in_channels: int = 3,
        # ── 兼容旧接口（不影响结构）──
        depth: int = 18,
        input_size: int = 32,
    ):
        super().__init__()

        def _ch(idx: int) -> int:
            """根据 slim_idx 决定第 idx 层是否应用 scale 缩放。"""
            base = self._BASE_CHANNELS[idx]
            return int(base * scale) if idx >= slim_idx else base

        trs = track_running_stats

        # ── Stem (idx=0) ─────────────────────────────────────────────────────
        stem_ch = _ch(0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, stem_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch, track_running_stats=trs),
            nn.ReLU(inplace=True),
        )

        # ── Residual stages (idx=1~4) ─────────────────────────────────────────
        self._inchannel = stem_ch   # 在 _make_layer 内滚动更新

        self.layer1 = self._make_layer(_ch(1), num_blocks=2, stride=1, trs=trs)
        self.layer2 = self._make_layer(_ch(2), num_blocks=2, stride=2, trs=trs)
        self.layer3 = self._make_layer(_ch(3), num_blocks=2, stride=2, trs=trs)
        self.layer4 = self._make_layer(_ch(4), num_blocks=2, stride=2, trs=trs)

        # ── 分类头 ────────────────────────────────────────────────────────────
        self.fc = nn.Linear(self._inchannel, num_classes)

    def _make_layer(
        self, out_ch: int, num_blocks: int, stride: int, trs: bool
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_ResidualBlock(self._inchannel, out_ch, s, trs))
            self._inchannel = out_ch
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc(out)

    def get_conv_layer_names(self) -> List[str]:
        return [
            name for name, m in self.named_modules()
            if isinstance(m, nn.Conv2d)
        ]


# ── 工厂函数，整个项目唯一的模型创建入口 ──────────────────────
DATASET_CONFIGS = {
    'cifar10':  {'num_classes': 10,  'in_channels': 3, 'input_size': 32},
    'cifar100': {'num_classes': 100, 'in_channels': 3, 'input_size': 32},
    'femnist':  {'num_classes': 62,  'in_channels': 1, 'input_size': 28},
    'mnist':    {'num_classes': 10,  'in_channels': 1, 'input_size': 28},
}


def build_model(
    model_name: str,
    dataset_name: str,
    scale: float = 1.0,
    slim_idx: int = 0,
    track_running_stats: bool = False,
) -> nn.Module:
    """
    统一模型工厂。

    BN 统一使用 track_running_stats=False（联邦学习推荐设置）。

    调用示例：
        model = build_model('vgg16', 'cifar10')
        model = build_model('vgg16', 'cifar10', scale=0.5, slim_idx=2)
        model = build_model('resnet18', 'cifar100', scale=0.71)
        model = build_model('resnet18', 'femnist')
    """
    dataset_key = dataset_name.lower()
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. "
                         f"Available: {list(DATASET_CONFIGS)}")

    cfg = DATASET_CONFIGS[dataset_key]
    name = model_name.lower()

    if name == 'vgg16':
        return VGG16(
            num_classes=cfg['num_classes'],
            scale=scale,
            slim_idx=slim_idx,
            track_running_stats=track_running_stats,
        )

    elif name in ('resnet18', 'resnet34'):
        # depth / input_size 通过兼容参数透传，不影响结构
        return ResNetFed(
            num_classes=cfg['num_classes'],
            scale=scale,
            slim_idx=slim_idx,
            track_running_stats=track_running_stats,
            in_channels=cfg['in_channels'],
            depth=int(name.replace('resnet', '')),
            input_size=cfg['input_size'],
        )

    else:
        raise ValueError(f"Unknown model: {model_name!r}. "
                         f"Supported: 'vgg16', 'resnet18', 'resnet34'")


def _count_parameters(model: nn.Module):
    """Return (total_params, trainable_params) as integers."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    import sys

    argv = sys.argv[1:]
    model_name  = argv[0] if len(argv) >= 1 else 'resnet18'
    dataset_name = argv[1] if len(argv) >= 2 else 'cifar10'

    show_convs = len(argv) >= 3 and argv[2].lower() in ('show-convs', 'convs', 'conv')

    model = build_model(model_name, dataset_name)
    print(model)

    total, trainable = _count_parameters(model)
    print(f"\nModel: {model_name} | Dataset: {dataset_name}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    if show_convs and hasattr(model, 'get_conv_layer_names'):
        convs = model.get_conv_layer_names()
        print(f"\nConvolution layers ({len(convs)}):")
        for n in convs:
            print('  ', n)