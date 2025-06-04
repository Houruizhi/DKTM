import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

def dilated_conv3x3(in_planes, out_planes, dilation, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)

class CondCRPBlock(nn.Module):
    def __init__(self, features, n_stages, normalizer, act=nn.ReLU()):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_stages):
            self.norms.append(normalizer(features))
            self.convs.append(conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.maxpool(path)
            path = self.convs[i](path)
            x = path + x
        return x

class CondRCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, normalizer, act=nn.ReLU()):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_norm'.format(i + 1, j + 1), normalizer(features))
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1),
                        conv3x3(features, features, stride=1, bias=False))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)
            x += residual
        return x


class CondMSFBlock(nn.Module):
    def __init__(self, in_planes, features, normalizer):
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True))
            self.norms.append(normalizer(in_planes[i]))

    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums

class CondRefineBlock(nn.Module):
    def __init__(self, in_planes, features, normalizer, act=nn.ReLU(), start=False, end=False):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, normalizer, act)
            )

        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, normalizer, act) # conv+norm

        self.msf = CondMSFBlock(in_planes, features, normalizer) # upsampling

        self.crp = CondCRPBlock(features, 2, normalizer, act) # maxpool

    def forward(self, hs, y, output_shape):
        assert isinstance(hs, tuple) or isinstance(hs, list)

        h = self.msf(hs, y, output_shape)

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        else:
            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            )

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output

class ConditionalResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, resample=None, act=nn.ReLU(),
                 normalization=None, adjust_padding=False, dilation=None):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
            else:
                self.conv1 = nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
                self.normalize2 = normalization(output_dim)
                self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)

    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        return shortcut + output

import math
class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        middle_feature = num_features//4
        self.gamma_conv = nn.Sequential(
            nn.Conv2d(num_features,middle_feature,1),
            nn.ReLU(True),
            nn.Conv2d(middle_feature,num_features,1),
        )
        self.beta_conv = nn.Sequential(
            nn.Conv2d(num_features,middle_feature,1),
            nn.ReLU(True),
            nn.Conv2d(middle_feature,num_features,1),
        )

        def weight_init_gamma(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(1/num_features, 0.02)
        def weight_init_beta(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)

        self.gamma_conv.apply(weight_init_gamma)
        self.beta_conv.apply(weight_init_beta)

        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)

    def forward(self, x, y):
        y = y.repeat(1, x.shape[1], 1, 1)
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        means = means.unsqueeze(-1).unsqueeze(-1)
        y = y * means
        gamma, beta = self.gamma_conv(y), self.beta_conv(y)
        out = gamma * h + beta
        return out

# Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

from .foundation_models import vit_mae
from .attentions import DeformFusion, CALayer
    
class MAERefineNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, fusion_type=0, img_size=(256,256)):
        super().__init__()
        ngf = 48
        self.begin_conv = nn.Conv2d(in_channels, ngf, 3, stride=1, padding=1)

        self.vit_mae = vit_mae.__dict__['vit_base_patch16'](num_classes=1000,global_pool=False,img_size=img_size)
        
        self.encoder0 = ConditionalResidualBlock(48, 96, resample='down', normalization=ConditionalInstanceNorm2dPlus)
        self.encoder1 = ConditionalResidualBlock(96, 192, resample='down', normalization=ConditionalInstanceNorm2dPlus)
        self.encoder2 = ConditionalResidualBlock(192, 384, resample='down', normalization=ConditionalInstanceNorm2dPlus)
        self.encoder3 = ConditionalResidualBlock(384, 768, resample='down', normalization=ConditionalInstanceNorm2dPlus)
        
        self.refine1 = CondRefineBlock([384, 384], 192, normalizer=ConditionalInstanceNorm2dPlus)
        self.refine2 = CondRefineBlock([192, 192], 96, normalizer=ConditionalInstanceNorm2dPlus)
        self.refine3 = CondRefineBlock([96, 96], 48, normalizer=ConditionalInstanceNorm2dPlus)
        self.refine4 = CondRefineBlock([48, 48], ngf, normalizer=ConditionalInstanceNorm2dPlus, end=True)

        self.normalizer = ConditionalInstanceNorm2dPlus(ngf)
        self.act = nn.ReLU(inplace=True)
        self.end_conv = nn.Conv2d(ngf, out_channels, 3, stride=1, padding=1)
        
        self.fusion_type = fusion_type
        if fusion_type == 0:
            self.fusion_layer = nn.Sequential(
                nn.InstanceNorm2d(768+768),
                nn.Conv2d(768+768,384,1,1,0)
            )
        else:
            self.deform_layer = DeformFusion(768,768,768)
            self.attn = CALayer(768+768)
            self.fusion_layer = nn.Sequential(
                nn.InstanceNorm2d(768+768),
                nn.Conv2d(768+768,384,1,1,0),
                nn.LeakyReLU()
            )

        self.conv_head = nn.Conv2d(in_channels, 3, 3, stride=1, padding=1)

    def _compute_cond_module(self, module, x, y):
        for m in module:
            x = m(x, y)
        return x

    def forward(self, x, y):
        '''
        g: guided image
        '''

        x_feature = self.begin_conv(x)
        f1 = self.encoder0(x_feature, y)
        f2 = self.encoder1(f1, y)
        f3 = self.encoder2(f2, y)
        f4 = self.encoder3(f3, y)
        
        x = self.normalize_channel(self.conv_head(x))
        x_vit = self.vit_mae.forward_features(x)[-1]
        
        B,L,C = x_vit.shape
        L = int(math.sqrt(L))
        x_vit = x_vit.view(B, L, L, C).permute(0,3,1,2)
        
        if self.fusion_type == 0:
            fusion = self.fusion_layer(torch.cat([f4,x_vit],dim=1))
        else:
            f4 = self.deform_layer(f4, x_vit)
            fusion = torch.cat([f4,x_vit],dim=1)
            fusion = fusion * self.attn(fusion)
            fusion = self.fusion_layer(fusion)
            
        ref2 = self.refine1([fusion, f3], y, f3.shape[2:])
        ref1 = self.refine2([ref2, f2], y, f2.shape[2:])
        ref0 = self.refine3([ref1, f1], y, f1.shape[2:])
        output = self.refine4([ref0, x_feature], y, x_feature.shape[2:])

        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)

        return output
    
    def normalize_channel(self, img):
        n,c,h,w = img.shape
        assert c == 3
        img_vec = img.view(n,c,-1)
        mean = img_vec.mean(-1).view(n,c,1,1)
        std = img_vec.std(-1).view(n,c,1,1)
        img = (img - mean) / (std + 1e-11)

        img = img * torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(img.device).view(1,3,1,1)
        img = img + torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(img.device).view(1,3,1,1)
        return img