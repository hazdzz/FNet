import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class FourierTransform(nn.Module):
    def __init__(self) -> None:
        super(FourierTransform, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.fft.fft2(input, dim=(-2, -1)).real
    

class FeedForward(nn.Module):
    def __init__(self, feat_dim, hid_dim, ffn_drop_prob) -> None:
        super(FeedForward, self).__init__()
        self.feat_dim = feat_dim
        self.linear_layer1 = nn.Linear(in_features=feat_dim, out_features=hid_dim, bias=True)
        self.linear_layer2 = nn.Linear(in_features=hid_dim, out_features=feat_dim, bias=True)
        self.gelu = nn.GELU()
        self.ffn_dropout = nn.Dropout(p=ffn_drop_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.linear_layer1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.linear_layer2.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, input: Tensor) -> Tensor:
        input_linear = self.linear_layer1(input)
        input_linear = self.gelu(input_linear)
        input_linear = self.ffn_dropout(input_linear)
        ffn_output = self.linear_layer2(input_linear)

        return ffn_output


class PostLayerNorm(nn.Module):
    def __init__(self, dim, func) -> None:
        super(PostLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.func = func
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.layernorm(self.func(input, **kwargs) + input)
    

class FNet(nn.Module):
    def __init__(self, args) -> None:
        super(FNet, self).__init__()
        self.fnet_block = nn.ModuleList([])
        for _ in range(args.num_block):
            self.fnet_block.append(nn.ModuleList([
                PostLayerNorm(args.embed_size, FourierTransform()),
                PostLayerNorm(args.embed_size, FeedForward(args.embed_size, args.hidden_size, args.ffn_drop_prob))
            ]))
    
    def forward(self, input: Tensor) -> Tensor:
        for fourier, ffn in self.fnet_block:
            input = fourier(input)
            input = ffn(input)
        
        return input