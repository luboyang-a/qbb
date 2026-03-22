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
        curr_dtype = linear_layer.weight.dtype
        fp_weight = linear_layer.weight.data
        qbb_tool = QBB_v1(k=k)
        bases, alphas, _ = qbb_tool.decompose(fp_weight.detach())
        bases, alphas = qbb_tool.upd(fp_weight, bases, alphas, steps=5)
        return cls(
            bases=torch.stack(bases).to(device),
            alphas=torch.stack(alphas).to(device).to(dtype=curr_dtype),
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features
        )
    
    def forward(self, x):
        W_q = torch.zeros(self.bases[0].shape, device=x.device, dtype=x.dtype)
        for i in range(len(self.alphas)):
            W_q = W_q + self.alphas[i].to(x.dtype) * self.bases[i].to(device=x.device, dtype=x.dtype)
        return torch.matmul(x, W_q.t())