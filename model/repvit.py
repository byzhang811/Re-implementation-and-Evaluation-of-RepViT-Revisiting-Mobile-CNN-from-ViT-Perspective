"""
Filename: model/repvit.py
Author: Jiatong
Date: 2025-11-01
Lines: 483
Description: RepViT model architecture implementation.
"""

import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite
from timm.models.vision_transformer import trunc_normal_
from timm.models import register_model


def align_channels_to_divisor(channel_num, divisor=8, min_channel=None):
    """
    Align channel number to be divisible by divisor.
    
    Args:
        channel_num: Input channel number
        divisor: Divisor to align to
        min_channel: Minimum channel number
        
    Returns:
        Aligned channel number
    """
    if min_channel is None:
        min_channel = divisor
    aligned = max(min_channel, int(channel_num + divisor / 2) // divisor * divisor)
    if aligned < 0.9 * channel_num:
        aligned += divisor
    return aligned


class Conv2d_BN(torch.nn.Sequential):
    """Convolution layer followed by BatchNorm."""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bn_init_weight=1, resolution=-10000):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            dilation: Convolution dilation
            groups: Number of groups for grouped convolution
            bn_init_weight: Initial weight for BatchNorm
            resolution: Unused parameter (for compatibility)
        """
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(out_channels))
        torch.nn.init.constant_(self.bn.weight, bn_init_weight)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """Fuse Conv2d and BatchNorm into a single Conv2d layer."""
        conv_layer, bn_layer = self._modules.values()
        weight_scale = bn_layer.weight / (bn_layer.running_var + bn_layer.eps)**0.5
        fused_weight = conv_layer.weight * weight_scale[:, None, None, None]
        fused_bias = bn_layer.bias - bn_layer.running_mean * bn_layer.weight / \
            (bn_layer.running_var + bn_layer.eps)**0.5
        fused_conv = torch.nn.Conv2d(
            fused_weight.size(1) * self.c.groups, fused_weight.size(0),
            fused_weight.shape[2:], stride=self.c.stride, padding=self.c.padding,
            dilation=self.c.dilation, groups=self.c.groups, device=conv_layer.weight.device)
        fused_conv.weight.data.copy_(fused_weight)
        fused_conv.bias.data.copy_(fused_bias)
        return fused_conv


class Residual(torch.nn.Module):
    """Residual connection wrapper with optional dropout."""
    
    def __init__(self, module, dropout_rate=0.):
        """
        Args:
            module: Module to wrap with residual connection
            dropout_rate: Dropout rate for stochastic depth
        """
        super().__init__()
        self.module = module
        self.dropout_rate = dropout_rate

    def forward(self, input_tensor):
        if self.training and self.dropout_rate > 0:
            dropout_mask = torch.rand(input_tensor.size(0), 1, 1, 1,
                                     device=input_tensor.device).ge_(self.dropout_rate).div(1 - self.dropout_rate).detach()
            return input_tensor + self.module(input_tensor) * dropout_mask
        else:
            return input_tensor + self.module(input_tensor)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.module, Conv2d_BN):
            fused_module = self.module.fuse()
            assert(fused_module.groups == fused_module.in_channels)
            identity_weight = torch.ones(fused_module.weight.shape[0], fused_module.weight.shape[1], 1, 1)
            identity_weight = torch.nn.functional.pad(identity_weight, [1, 1, 1, 1])
            fused_module.weight += identity_weight.to(fused_module.weight.device)
            return fused_module
        elif isinstance(self.module, torch.nn.Conv2d):
            fused_module = self.module
            assert(fused_module.groups != fused_module.in_channels)
            identity_weight = torch.ones(fused_module.weight.shape[0], fused_module.weight.shape[1], 1, 1)
            identity_weight = torch.nn.functional.pad(identity_weight, [1, 1, 1, 1])
            fused_module.weight += identity_weight.to(fused_module.weight.device)
            return fused_module
        else:
            return self


class RepVGGDW(torch.nn.Module):
    """RepVGG-style depthwise convolution block."""
    
    def __init__(self, embed_dim) -> None:
        """
        Args:
            embed_dim: Embedding dimension (channels)
        """
        super().__init__()
        self.conv3x3 = Conv2d_BN(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)
        self.conv1x1 = torch.nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, groups=embed_dim)
        self.embed_dim = embed_dim
        self.bn = torch.nn.BatchNorm2d(embed_dim)

    def forward(self, input_tensor):
        return self.bn((self.conv3x3(input_tensor) + self.conv1x1(input_tensor)) + input_tensor)

    @torch.no_grad()
    def fuse(self):
        fused_conv3x3 = self.conv3x3.fuse()
        conv1x1 = self.conv1x1

        conv3x3_weight = fused_conv3x3.weight
        conv3x3_bias = fused_conv3x3.bias
        conv1x1_weight = conv1x1.weight
        conv1x1_bias = conv1x1.bias

        conv1x1_weight = torch.nn.functional.pad(conv1x1_weight, [1, 1, 1, 1])
        identity_weight = torch.nn.functional.pad(
            torch.ones(conv1x1_weight.shape[0], conv1x1_weight.shape[1], 1, 1, device=conv1x1_weight.device),
            [1, 1, 1, 1])

        combined_weight = conv3x3_weight + conv1x1_weight + identity_weight
        combined_bias = conv3x3_bias + conv1x1_bias

        fused_conv3x3.weight.data.copy_(combined_weight)
        fused_conv3x3.bias.data.copy_(combined_bias)

        bn_layer = self.bn
        bn_weight_scale = bn_layer.weight / (bn_layer.running_var + bn_layer.eps)**0.5
        final_weight = fused_conv3x3.weight * bn_weight_scale[:, None, None, None]
        final_bias = bn_layer.bias + (fused_conv3x3.bias - bn_layer.running_mean) * bn_layer.weight / \
            (bn_layer.running_var + bn_layer.eps)**0.5
        fused_conv3x3.weight.data.copy_(final_weight)
        fused_conv3x3.bias.data.copy_(final_bias)
        return fused_conv3x3


class RepViTBlock(nn.Module):
    """Basic building block of RepViT architecture."""
    
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size, stride, use_se, use_hs):
        """
        Args:
            input_channels: Input channel number
            hidden_channels: Hidden dimension (expansion)
            output_channels: Output channel number
            kernel_size: Convolution kernel size
            stride: Stride value (1 or 2)
            use_se: Whether to use Squeeze-Excitation
            use_hs: Whether to use h-swish activation
        """
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]
        self.has_identity = stride == 1 and input_channels == output_channels
        assert(hidden_channels == 2 * input_channels)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(input_channels, input_channels, kernel_size, stride, (kernel_size - 1) // 2, groups=input_channels),
                SqueezeExcite(input_channels, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(input_channels, output_channels, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(output_channels, 2 * output_channels, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                Conv2d_BN(2 * output_channels, output_channels, 1, 1, 0, bn_init_weight=0),
            ))
        else:
            assert(self.has_identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(input_channels),
                SqueezeExcite(input_channels, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(input_channels, hidden_channels, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                Conv2d_BN(hidden_channels, output_channels, 1, 1, 0, bn_init_weight=0),
            ))

    def forward(self, input_tensor):
        return self.channel_mixer(self.token_mixer(input_tensor))


class BN_Linear(torch.nn.Sequential):
    """BatchNorm followed by Linear layer."""
    
    def __init__(self, in_features, out_features, use_bias=True, std=0.02):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            use_bias: Whether to use bias in linear layer
            std: Standard deviation for weight initialization
        """
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(in_features))
        self.add_module('l', torch.nn.Linear(in_features, out_features, bias=use_bias))
        trunc_normal_(self.l.weight, std=std)
        if use_bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn_layer, linear_layer = self._modules.values()
        weight_scale = bn_layer.weight / (bn_layer.running_var + bn_layer.eps)**0.5
        bias_offset = bn_layer.bias - self.bn.running_mean * \
            self.bn.weight / (bn_layer.running_var + bn_layer.eps)**0.5
        fused_weight = linear_layer.weight * weight_scale[None, :]
        if linear_layer.bias is None:
            fused_bias = bias_offset @ self.l.weight.T
        else:
            fused_bias = (linear_layer.weight @ bias_offset[:, None]).view(-1) + self.l.bias
        fused_linear = torch.nn.Linear(fused_weight.size(1), fused_weight.size(0), device=linear_layer.weight.device)
        fused_linear.weight.data.copy_(fused_weight)
        fused_linear.bias.data.copy_(fused_bias)
        return fused_linear


class Classfier(nn.Module):
    """Classification head for RepViT."""
    
    def __init__(self, feature_dim, num_classes):
        """
        Args:
            feature_dim: Input feature dimension
            num_classes: Number of output classes
        """
        super().__init__()
        self.classifier = BN_Linear(feature_dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, input_tensor):
        return self.classifier(input_tensor)

    @torch.no_grad()
    def fuse(self):
        return self.classifier.fuse()


class RepViT(nn.Module):
    """RepViT model architecture."""
    
    def __init__(self, configs, num_classes=1000, distillation=False):
        """
        Args:
            configs: List of block configurations [kernel, expansion, channels, se, hs, stride]
            num_classes: Number of classification classes
            distillation: Unused parameter (for compatibility)
        """
        super(RepViT, self).__init__()
        self.configs = configs

        initial_channels = self.configs[0][2]
        stem_layer = torch.nn.Sequential(
            Conv2d_BN(3, initial_channels // 2, 3, 2, 1),
            torch.nn.GELU(),
            Conv2d_BN(initial_channels // 2, initial_channels, 3, 2, 1))
        layer_list = [stem_layer]

        block_type = RepViTBlock
        current_channels = initial_channels
        for kernel, expansion, channels, se_flag, hs_flag, stride_val in self.configs:
            output_channels = align_channels_to_divisor(channels, 8)
            expanded_channels = align_channels_to_divisor(current_channels * expansion, 8)
            layer_list.append(block_type(current_channels, expanded_channels, output_channels,
                                        kernel, stride_val, se_flag, hs_flag))
            current_channels = output_channels
        self.features = nn.ModuleList(layer_list)
        self.classifier = Classfier(current_channels, num_classes)

    def forward(self, input_tensor):
        for layer in self.features:
            input_tensor = layer(input_tensor)
        input_tensor = torch.nn.functional.adaptive_avg_pool2d(input_tensor, 1).flatten(1)
        return self.classifier(input_tensor)


@register_model
def repvit_m0_6(pretrained=False, num_classes=1000, distillation=False):
    """
    Create RepViT m0_6 model.
    
    Args:
        pretrained: Whether to load pretrained weights
        num_classes: Number of classification classes
        distillation: Unused parameter (for compatibility)
        
    Returns:
        RepViT model instance
    """
    config_list = [
        [3, 2, 40, 1, 0, 1],
        [3, 2, 40, 0, 0, 1],
        [3, 2, 80, 0, 0, 2],
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 160, 0, 1, 2],
        [3, 2, 160, 1, 1, 1],
        [3, 2, 160, 0, 1, 1],
        [3, 2, 160, 1, 1, 1],
        [3, 2, 160, 0, 1, 1],
        [3, 2, 160, 1, 1, 1],
        [3, 2, 160, 0, 1, 1],
        [3, 2, 160, 1, 1, 1],
        [3, 2, 160, 0, 1, 1],
        [3, 2, 160, 0, 1, 1],
        [3, 2, 320, 0, 1, 2],
        [3, 2, 320, 1, 1, 1],
    ]
    return RepViT(config_list, num_classes=num_classes, distillation=False)


def build_m0_9_configs(stage_depths=(3, 4, 16, 3), stage_widths=(48, 96, 192, 384)):
    kernel_size, expansion_ratio = 3, 2
    config_list = []
    for stage_idx, (width, depth) in enumerate(zip(stage_widths, stage_depths)):
        width = align_channels_to_divisor(width, 8)
        use_hswish = 1 if stage_idx >= 2 else 0

        if stage_idx > 0:
            config_list.append([kernel_size, expansion_ratio, width, 0, use_hswish, 2])
            remaining_blocks = depth - 1
        else:
            remaining_blocks = depth

        for block_idx in range(remaining_blocks):
            use_se = 1 if (block_idx % 2 == 0) else 0
            config_list.append([kernel_size, expansion_ratio, width, use_se, use_hswish, 1])
    return config_list


def scale_channel_widths(base_widths=(48, 96, 192, 384), scale_factor=1.0):
    """
    Scale channel widths proportionally.
    
    Args:
        base_widths: Base channel widths
        scale_factor: Scaling factor
        
    Returns:
        List of scaled and aligned channel widths
    """
    return [align_channels_to_divisor(int(round(w * scale_factor)), 8) for w in base_widths]


@register_model
def repvit_m0_9_w088(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with width scaled to 88%."""
    scaled_widths = scale_channel_widths(scale_factor=0.88)
    config_list = build_m0_9_configs(depths=(3, 4, 16, 3), widths=scaled_widths)
    return RepViT(config_list, num_classes=num_classes, distillation=False)


@register_model
def repvit_m0_9_w112(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with width scaled to 112%."""
    scaled_widths = scale_channel_widths(scale_factor=1.12)
    config_list = build_m0_9_configs(depths=(3, 4, 16, 3), widths=scaled_widths)
    return RepViT(config_list, num_classes=num_classes, distillation=False)


@register_model
def repvit_m0_9_w125(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with width scaled to 125%."""
    scaled_widths = scale_channel_widths(scale_factor=1.25)
    config_list = build_m0_9_configs(depths=(3, 4, 16, 3), widths=scaled_widths)
    return RepViT(config_list, num_classes=num_classes, distillation=False)


@register_model
def repvit_m0_9_d22142(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with stage depths [2, 2, 14, 2]."""
    config_list = build_m0_9_configs(depths=(2, 2, 14, 2))
    return RepViT(config_list, num_classes=num_classes, distillation=False)


@register_model
def repvit_m0_9_d22182(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with stage depths [2, 2, 18, 2]."""
    config_list = build_m0_9_configs(depths=(2, 2, 18, 2))
    return RepViT(config_list, num_classes=num_classes, distillation=False)


@register_model
def repvit_m0_9_d24122(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with stage depths [2, 4, 12, 2]."""
    config_list = build_m0_9_configs(depths=(2, 4, 12, 2))
    return RepViT(config_list, num_classes=num_classes, distillation=False)


def get_m0_9_baseline_configs():
    """Get baseline configuration for m0_9 model."""
    return [
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 96, 0, 0, 2],
        [3, 2, 96, 1, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 192, 0, 1, 2],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 384, 0, 1, 2],
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 0, 1, 1],
    ]


def modify_kernel_sizes(config_list, k_stage1=3, k_stage2=3, k_stage3=3, k_stage4=3):
    """
    Modify kernel sizes for different stages.
    
    Args:
        config_list: Configuration list to modify
        k_stage1: Kernel size for stage 1
        k_stage2: Kernel size for stage 2
        k_stage3: Kernel size for stage 3
        k_stage4: Kernel size for stage 4
        
    Returns:
        Modified configuration list
    """
    for config_item in config_list:
        channel = config_item[2]
        if channel == 48:
            config_item[0] = k_stage1
        elif channel == 96:
            config_item[0] = k_stage2
        elif channel == 192:
            config_item[0] = k_stage3
        elif channel == 384:
            config_item[0] = k_stage4
    return config_list


@register_model
def repvit_m0_9_k5555(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with kernel size 5 for all stages."""
    config_list = modify_kernel_sizes(get_m0_9_baseline_configs(), 5, 5, 5, 5)
    return RepViT(config_list, num_classes=num_classes, distillation=False)


@register_model
def repvit_m0_9_k3355(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with kernel sizes [3, 3, 5, 5] for stages."""
    config_list = modify_kernel_sizes(get_m0_9_baseline_configs(), 3, 3, 5, 5)
    return RepViT(config_list, num_classes=num_classes, distillation=False)


@register_model
def repvit_m0_9_k3353(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with kernel sizes [3, 3, 5, 3] for stages."""
    config_list = modify_kernel_sizes(get_m0_9_baseline_configs(), 3, 3, 5, 3)
    return RepViT(config_list, num_classes=num_classes, distillation=False)


def apply_se_strategy(config_list, strategy="alt", start_with_se=True):
    """
    Apply Squeeze-Excitation strategy to configuration list.
    
    Args:
        config_list: Configuration list to modify
        strategy: SE strategy ("none", "alt", "all")
        start_with_se: Whether to start with SE enabled (for alternating strategy)
        
    Returns:
        Modified configuration list
    """
    result_list = []
    se_enabled = bool(start_with_se)
    for kernel, expansion, channels, se_current, hswish, stride in config_list:
        if strategy == "none":
            se_new = 0
        elif strategy == "all":
            se_new = 1 if stride != 2 else 0
        elif strategy == "alt":
            if stride == 2:
                se_new = 0
                se_enabled = bool(start_with_se)
            else:
                se_new = 1 if se_enabled else 0
                se_enabled = not se_enabled
        else:
            raise ValueError(f"Unknown SE strategy: {strategy}")
        result_list.append([kernel, expansion, channels, se_new, hswish, stride])
    return result_list


@register_model
def repvit_m0_9_se_none(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with SE disabled for all blocks."""
    config_list = apply_se_strategy(get_m0_9_baseline_configs(), strategy="none")
    return RepViT(config_list, num_classes=num_classes, distillation=False)


@register_model
def repvit_m0_9_se_alt(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with alternating SE pattern."""
    config_list = apply_se_strategy(get_m0_9_baseline_configs(), strategy="alt", start_with_se=True)
    return RepViT(config_list, num_classes=num_classes, distillation=False)


@register_model
def repvit_m0_9_se_all(pretrained=False, num_classes=1000, distillation=False):
    """RepViT m0_9 with SE enabled for all blocks (except downsampling)."""
    config_list = apply_se_strategy(get_m0_9_baseline_configs(), strategy="all")
    return RepViT(config_list, num_classes=num_classes, distillation=False)


@register_model
def repvit_m0_9(pretrained=False, num_classes=1000, distillation=False):
    """Create RepViT m0_9 baseline model."""
    config_list = [
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 96, 0, 0, 2],
        [3, 2, 96, 1, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 192, 0, 1, 2],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 384, 0, 1, 2],
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 0, 1, 1]
    ]
    return RepViT(config_list, num_classes=num_classes, distillation=False)
