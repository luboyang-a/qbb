import torch
import torch.nn as nn
from qbb_model import QBBLinear
import torch.nn.functional as F
from tqdm import tqdm

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

class FeatureHook:

    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output[0] if isinstance(output, tuple) else output

    def remove(self):
        self.hook.remove()
        self.features = None

class QBBCalibrator:

    def __init__(self, student, teacher, tokenizer, s1=1.0, s2=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self.tokenizer = tokenizer
        self.s1 = s1
        self.s2 = s2
        self.s_hooks = []
        self.t_hooks = []

    def _setup_hooks(self):
        for s_layer, t_layer in zip(self.student.model.layers, self.teacher.model.layers):
            self.s_hooks.append(FeatureHook(s_layer))
            self.t_hooks.append(FeatureHook(t_layer))

    def _cleanup_hooks(self):
        for h in self.s_hooks + self.t_hooks:
            h.remove()
        self.s_hooks = []
        self.t_hooks = []

    def generate_synthetic_data(self, num_samples=64, seq_len=128):
        print("generating data...\n")
        self.teacher.eval()
        calib_ids = []
        with torch.no_grad():
            for _ in range(num_samples):
                start_id = torch.randint(0, self.tokenizer.vocab_size, (1, 1)).to(self.device)
                out = self.teacher.generate(
                    start_id, max_length=seq_len, do_sample=True,
                    temperature=1.0, pad_token_id=self.tokenizer.eos_token_id
                )
                calib_ids.append(out)
        return torch.cat(calib_ids, dim=0)
            
    def calibrate(self, epochs=3, lr=1e-4, batch_size=4):
        calib_data = self.generate_synthetic_data()
        trainable_params = [p for n, p in self.student.named_parameters() if "alphas" in n]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        self._setup_hooks()
        print("start calibrating model...\n")
        self.student.train()
        self.teacher.eval()
        num_batches = (calib_data.size(0) + batch_size - 1) // batch_size
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, calib_data.size(0), batch_size):
                batch = calib_data[i: i + batch_size]
                optimizer.zero_grad()
                with torch.no_grad():
                    t_output = self.teacher(batch)
                s_output = self.student(batch)
                l_mse = F.mse_loss(s_output.logits, t_output.logits)
                l_feat = sum([F.mse_loss(sh.features, th.features) for sh, th in zip(self.s_hooks, self.t_hooks)])
                loss = self.s1 * l_mse + self.s2 * l_feat
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1} | Loss: {total_loss / num_batches:.6f}")
        self._cleanup_hooks()
        print("Complated\n")