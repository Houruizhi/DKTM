import torch
import torch.nn as nn
from .weight import weights_init_kaiming
from .norm_layer import get_norm_layer

class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)

        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        h = self.instance_norm(x)

        gamma, beta = self.embed(y).chunk(2, dim=-1)
        out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)

        return out

class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_layers, num_features, num_classes, kernel_size=3, residual=True, bias=False):
        super(DnCNN, self).__init__()
        kernel_size = kernel_size
        padding = kernel_size//2
        features = num_features
        in_channels = in_channels
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, padding=padding, kernel_size=kernel_size, bias=bias),
            nn.ReLU(inplace=True)
            )
        
        self.cnns = nn.ModuleList(
            [nn.Conv2d(in_channels=features, out_channels=features, padding=padding,
                kernel_size=kernel_size, bias=bias) for _ in range(num_of_layers-2)]
            )

        self.norms = nn.ModuleList(
            [ConditionalInstanceNorm2d(features, num_classes) for _ in range(num_of_layers-2)]
            )
        
        self.acts = nn.ModuleList(
            [nn.LeakyReLU(inplace=True) for _ in range(num_of_layers-2)]
            )
        
        self.tail = nn.Conv2d(in_channels=features, out_channels=out_channels, padding=padding,
            kernel_size=kernel_size, bias=bias)

        self.num_of_layers = num_of_layers
        
    def forward(self, inputs, t):
        x = self.head(inputs)
        
        for k in range(self.num_of_layers-2):
            x = self.norms[k](x,t)
            x = self.acts[k](x)
            x = self.cnns[k](x)
        last_feature = x
        return self.tail(last_feature)