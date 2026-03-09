import torch
import torch.nn as nn
from qbb_model import QBBLinear

def qbb_replace(model, k=4, verbose=True):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name != "lm_head":
            if verbose:
                print(f" replace: {name} | in: {child.in_features} -> out: {child.out_features}")
            new_layer = QBBLinear.from_linear(child, k=k)
            setattr(model, name, new_layer)
        else:
            qbb_replace(child, k=k, verbose=verbose)
    return model