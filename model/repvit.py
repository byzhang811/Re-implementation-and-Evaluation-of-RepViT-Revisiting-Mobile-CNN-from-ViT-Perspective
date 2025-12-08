import torch.nn as nn

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

from timm.models.layers import SqueezeExcite

import torch

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

from timm.models.vision_transformer import trunc_normal_
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier

class RepViT(nn.Module):
    def __init__(self, cfgs, num_classes=1000, distillation=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                           Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self.classifier = Classfier(output_channel, num_classes, distillation)
        
    def forward(self, x):
        # x = self.features(x)
        for f in self.features:
            x = f(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        return x

from timm.models import register_model


@register_model
def repvit_m0_6(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        [3,   2,  40, 1, 0, 1],
        [3,   2,  40, 0, 0, 1],
        [3,   2,  80, 0, 0, 2],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  160, 0, 1, 2],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 320, 0, 1, 2],
        [3,   2, 320, 1, 1, 1],
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

##########################################################################################
##########################################################################################

def _build_cfgs_m0_9(depths=(3, 4, 16, 3), widths=(48, 96, 192, 384)):
    """
    depths: Number of blocks in each stage (including the downsampling block of that stage; stage0 has no downsampling)
    widths: Channel width sequence for each stage
    Other settings follow m0_9 baseline: k=3, t=2; SE alternates; stage2/3 use h-swish.
    """
    k, t = 3, 2
    cfgs = []
    for si, (c, d) in enumerate(zip(widths, depths)):
        c = _make_divisible(c, 8)
        use_hs = 1 if si >= 2 else 0

        if si > 0:
            cfgs.append([k, t, c, 0, use_hs, 2])  # 下采样块（SE=0, s=2）
            remain = d - 1
        else:
            remain = d

        for i in range(remain):
            use_se = 1 if (i % 2 == 0) else 0
            cfgs.append([k, t, c, use_se, use_hs, 1])
    return cfgs


def _scale_widths(base=(48, 96, 192, 384), scale=1.0):
    """Scale channels proportionally and align to multiples of 8."""
    return [_make_divisible(int(round(c * scale)), 8) for c in base]

@register_model
def repvit_m0_9_w088(pretrained=False, num_classes=1000, distillation=False):
    """Width -12%: 48,96,192,384 → 42,84,168,336 (after alignment to 8)"""
    widths = _scale_widths(scale=0.88)
    cfgs = _build_cfgs_m0_9(depths=(3, 4, 16, 3), widths=widths)
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m0_9_w112(pretrained=False, num_classes=1000, distillation=False):
    """Width +12%: 48,96,192,384 → ~54,108,216,432 (after alignment to 8)"""
    widths = _scale_widths(scale=1.12)
    cfgs = _build_cfgs_m0_9(depths=(3, 4, 16, 3), widths=widths)
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m0_9_w125(pretrained=False, num_classes=1000, distillation=False):
    """Width +25%: 48,96,192,384 → 60,120,240,480 (after alignment to 8)"""
    widths = _scale_widths(scale=1.25)
    cfgs = _build_cfgs_m0_9(depths=(3, 4, 16, 3), widths=widths)
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m0_9_d22142(pretrained=False, num_classes=1000, distillation=False):
    """
    RepViT m0_9 with stage depths replaced to [2, 2, 14, 2]
    """
    cfgs = _build_cfgs_m0_9(depths=(2, 2, 14, 2))
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

@register_model
def repvit_m0_9_d22182(pretrained=False, num_classes=1000, distillation=False):
    """
    Depth = [2, 2, 18, 2]: Place extra computation in Stage-3 (14×14, C=192),
    Goal: Deepen at smaller resolution to gain higher accuracy while controlling latency growth.
    """
    cfgs = _build_cfgs_m0_9(depths=(2, 2, 18, 2))
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

@register_model
def repvit_m0_9_d24122(pretrained=False, num_classes=1000, distillation=False):
    """
    Depth = [2, 4, 12, 2]: Deepen Stage-2 (28×28, C=96), slightly reduce Stage-3,
    Goal: Test the impact of medium-scale texture/edge representation on accuracy, and reduce Stage-4 high channel cost.
    """
    cfgs = _build_cfgs_m0_9(depths=(2, 4, 12, 2))
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

##########################################################################################

def _repvit_m0_9_base_cfgs():
    # k, t, c, use_se, use_hs, s  (baseline)
    return [
        [3,2,48, 1,0,1],
        [3,2,48, 0,0,1],
        [3,2,48, 0,0,1],
        [3,2,96, 0,0,2],
        [3,2,96, 1,0,1],
        [3,2,96, 0,0,1],
        [3,2,96, 0,0,1],
        [3,2,192,0,1,2],
        [3,2,192,1,1,1],
        [3,2,192,0,1,1],
        [3,2,192,1,1,1],
        [3,2,192,0,1,1],
        [3,2,192,1,1,1],
        [3,2,192,0,1,1],
        [3,2,192,1,1,1],
        [3,2,192,0,1,1],
        [3,2,192,1,1,1],
        [3,2,192,0,1,1],
        [3,2,192,1,1,1],
        [3,2,192,0,1,1],
        [3,2,192,1,1,1],
        [3,2,192,0,1,1],
        [3,2,192,0,1,1],
        [3,2,384,0,1,2],
        [3,2,384,1,1,1],
        [3,2,384,0,1,1],
    ]

def _set_kernel_per_stage(cfgs, k_s1=3, k_s2=3, k_s3=3, k_s4=3):
    # Determine stage by output channel c: 48/96/192/384
    for it in cfgs:
        c = it[2]
        if c == 48:
            it[0] = k_s1
        elif c == 96:
            it[0] = k_s2
        elif c == 192:
            it[0] = k_s3
        elif c == 384:
            it[0] = k_s4
    return cfgs


@register_model
def repvit_m0_9_k5555(pretrained=False, num_classes=1000, distillation=False):
    """
    All stages use k=5
    """
    cfgs = _set_kernel_per_stage(_repvit_m0_9_base_cfgs(), 5, 5, 5, 5)
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m0_9_k3355(pretrained=False, num_classes=1000, distillation=False):
    """
    Stage1/2 use k=3, Stage3/4 use k=5
    """
    cfgs = _set_kernel_per_stage(_repvit_m0_9_base_cfgs(), 3, 3, 5, 5)
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m0_9_k3353(pretrained=False, num_classes=1000, distillation=False):
    """
    Only Stage3 uses k=5, others remain k=3
    """
    cfgs = _set_kernel_per_stage(_repvit_m0_9_base_cfgs(), 3, 3, 5, 3)
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

##########################################################################################

def _apply_se_policy(cfgs, policy="alt", alt_start=True):
    """
    policy: "none" / "alt" / "all"
    alt_start: Under alternating policy, whether the first s=1 block of each stage starts with SE=1
    Convention: All s=2 (downsampling) blocks keep SE=0
    """
    out = []
    use = bool(alt_start)
    for k, t, c, se, hs, s in cfgs:
        if policy == "none":
            se_new = 0
        elif policy == "all":
            se_new = 1 if s != 2 else 0              # 下采样仍置 0
        elif policy == "alt":
            if s == 2:
                se_new = 0
                use = bool(alt_start)                 # 新 stage 重置交替
            else:
                se_new = 1 if use else 0
                use = not use
        else:
            raise ValueError(f"Unknown SE policy: {policy}")
        out.append([k, t, c, se_new, hs, s])
    return out

@register_model
def repvit_m0_9_se_none(pretrained=False, num_classes=1000, distillation=False):
    """SE all off"""
    cfgs = _apply_se_policy(_repvit_m0_9_base_cfgs(), policy="none")
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

@register_model
def repvit_m0_9_se_alt(pretrained=False, num_classes=1000, distillation=False):
    """SE alternating (s=1 blocks in each stage alternate 1,0,1,0,...; s=2 blocks fixed at 0)"""
    cfgs = _apply_se_policy(_repvit_m0_9_base_cfgs(), policy="alt", alt_start=True)
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

@register_model
def repvit_m0_9_se_all(pretrained=False, num_classes=1000, distillation=False):
    """SE all on (but s=2 downsampling blocks still set to 0, maintaining consistent boundary design with most implementations)"""
    cfgs = _apply_se_policy(_repvit_m0_9_base_cfgs(), policy="all")
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

##########################################################################################
##########################################################################################

@register_model
def repvit_m0_9(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  48, 1, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  96, 0, 0, 2],
        [3,   2,  96, 1, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  192, 0, 1, 2],
        [3,   2,  192, 1, 1, 1],
        [3,   2,  192, 0, 1, 1],
        [3,   2,  192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 384, 0, 1, 2],
        [3,   2, 384, 1, 1, 1],
        [3,   2, 384, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)