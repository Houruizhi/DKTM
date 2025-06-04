import torch
import torch.nn as nn
import torch.nn.functional as F

class blockNL(torch.nn.Module):
    def __init__(self, channels, middle_channles=32, fs=15):
        super(blockNL, self).__init__()
        self.channels = channels
        self.fs = fs
        self.softmax = nn.Softmax(dim=-1)

        self.t = nn.Conv2d(in_channels=self.channels, out_channels=middle_channles, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(in_channels=self.channels, out_channels=middle_channles, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):

        x_fea = x

        theta = self.t(x_fea).permute(0, 2, 3, 1)#.contiguous()#[b, c, h, w]#[b, h, w,c]
        theta = torch.unsqueeze(theta, dim=-2)  # [b, h, w, 1, c]
        # print(theta.size())

        phi = self.p(x_fea)#[b, c, h, w]
        b, c, h, w = phi.size()
        phi_patches = F.unfold(phi, self.fs, padding=self.fs // 2)#[b, c*fs*fs, hw]
        phi_patches = phi_patches.view(b, c, self.fs * self.fs, -1)#[b, c, fs*fs, hw]
        phi_patches = phi_patches.view(b, c, self.fs * self.fs, h, w)  #[b, c, fs*fs, h, w]
        phi_patches = phi_patches.permute(0, 3, 4, 1, 2)#.contiguous()#[b, h, w, c, fs*fs]
        # print(phi_patches.size())

        att = torch.matmul(theta, phi_patches)# [b, h, w, 1, fs*fs]
        att = self.softmax(att)# [b, h, w, 1, fs*fs]
        # print(att.size())

        b, c, h, w = x_fea.size()
        g = self.g(x_fea) #[b, 3, h, w]
        g_patches = F.unfold(g, self.fs, padding=self.fs // 2)#[b, 3*fs*fs, hw]
        g_patches = g_patches.view(b, c, self.fs * self.fs, -1)#[b, 3, fs*fs, hw]
        g_patches = g_patches.view(b, c, self.fs * self.fs, h, w)#[b, 3, fs*fs, h, w]
        g_patches = g_patches.permute(0, 3, 4, 2, 1)#.contiguous()#[b, h, w, fs*fs, 3]
        # print(g_patches.size())

        out_x = torch.matmul(att, g_patches)  # [1, h, w, 1, c]
        out_x = torch.squeeze(out_x, dim=-2)# [1, h, w, 3]
        out_x = out_x.permute(0, 3, 1, 2)#.contiguous()

        return self.w(out_x) + x
    
class SALayer(nn.Module):
    def __init__(self):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
                nn.Conv2d(2, 1, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, 3, padding=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        x_pool = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        y = self.conv_du(x_pool)
        return x * y
    
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x) + self.max_pool(x)
        y = self.conv_du(y)
        return x * y
    
class CBAM(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CBAM, self).__init__()
        self.attention_SALayer = SALayer()
        self.attention_CALayer = CALayer(channel, reduction=reduction)
        self.conv = nn.Sequential(*[
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU()
        ])
        
    def forward(self, x):
        x = self.conv(x)
        x = self.attention_CALayer(x)
        x = self.attention_SALayer(x)
        return x
    
class CBAM_FFT(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CBAM_FFT, self).__init__()
        self.conv_real = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_imag = nn.Conv2d(channel, channel, 3, padding=1)

        self.conv3 = nn.Conv2d(4, 1, 3, padding=1)
        
        self.attention_CALayer = CALayer(channel, reduction=reduction)
        
    def forward(self, x):

        q = torch.cat([x.unsqueeze(-1), torch.zeros_like(x).unsqueeze(-1)], dim=-1)
        q = torch.fft.fft2(torch.view_as_complex(q))

        r = self.conv_real(q.real)
        i = self.conv_imag(q.imag)

        r_pool = torch.cat((torch.max(r,1)[0].unsqueeze(1), torch.mean(r,1).unsqueeze(1)), dim=1)
        i_pool = torch.cat((torch.max(i,1)[0].unsqueeze(1), torch.mean(i,1).unsqueeze(1)), dim=1)

        m = self.conv3(torch.cat([r_pool, i_pool], dim=1))
        m = torch.sigmoid(m)

        r_weighted = (m * r)
        i_weighted = (m * i)
        x = torch.cat([r_weighted.unsqueeze(-1), i_weighted.unsqueeze(-1)], dim=-1)
        x = torch.fft.ifft2(torch.view_as_complex(x)).real

        x = self.attention_CALayer(x)
        return x