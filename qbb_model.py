import torch
import torch.nn as nn
from qbb_core import QBB_v1

class QBBLinear(nn.Module):

    def __init__(self, bases, alphas, in_features, out_features):
        super().__init__()
        self.register_buffer('bases', bases.to(torch.int8))
        self.alphas = nn.Parameter(alphas)
        self.in_features = in_features
        self.out_features = out_features

    @classmethod
    def from_linear(cls, linear_layer, k=4):
        device = linear_layer.weight.device
        fp_weight = linear_layer.weight.data
        qbb_tool = QBB_v1(k=k)
        bases, alphas, _ = qbb_tool.decompose(fp_weight.detach())
        bases, alphas = qbb_tool.upd(fp_weight, bases, alphas, steps=3)
        return cls(
            bases=torch.stack(bases).to(device),
            alphas=torch.stack(alphas).to(device),
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features
        )
    
    def forward(self, x):
        W_q = torch.sum(self.alphas * self.bases.to(device=x.device, dtype=x.dtype), dim=0)
        return torch.matmul(x, W_q.t())