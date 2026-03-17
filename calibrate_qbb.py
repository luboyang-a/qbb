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
        print(f"start to generate synthetic data...\n")
        self.teacher.eval()
        self.student.eval()
        candidates = []
        num_candidates = 2 * num_samples
        vocab_size = self.teacher.config.vocab_size
        start_ids = torch.randint(0, vocab_size, (num_candidates, 1)).to(self.device)
        with torch.no_grad():
            for i in range(num_candidates):
                out = self.teacher.generate(
                    start_ids[i:i+1],
                    max_length=seq_len,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                t_logits = self.teacher(out).logits
                s_logits = self.student(out).logits
                score = F.mse_loss(s_logits, t_logits).item()
                candidates.append((score, out))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1].cpu() for c in candidates[:num_samples]]
            
    def calibrate(self, epochs=3, lr=1e-4, batch_size=1):
        original_layers = [layer for layer in self.student.model.layers]
        calib_data = self.generate_synthetic_data()
        trainable_params = [p for n, p in self.student.named_parameters() if "alphas" in n]
        optimizer = torch.optim.Adam(trainable_params, lr=lr, eps=1e-6)
        self._setup_hooks()
        for epoch in range(epochs):
            self.student.train()
            total_loss = 0
            replace_prob = 0.5 * max(0, 1 - epoch / (epochs * 0.8))
            if epoch > 0 and epoch % 5 == 0:
                calib_data = self.generate_synthetic_data()
            pbar = tqdm(enumerate(calib_data), total=len(calib_data), desc=f"Epoch {epoch + 1}")
            for i, batch in pbar:
                for idx in range(len(self.student.model.layers)):
                    if torch.rand(1).item() < replace_prob:
                        self.student.model.layers[idx] = self.teacher.model.layers[idx]
                    else:
                        self.student.model.layers[idx] = original_layers[idx]
                batch = batch.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    t_output = self.teacher(batch)
                s_output = self.student(batch)
                l_mse = F.mse_loss(s_output.logits, t_output.logits)
                l_feat = sum([F.mse_loss(sh.features, th.features) for sh, th in zip(self.s_hooks, self.t_hooks)])
                loss = self.s1 * l_mse + self.s2 * l_feat
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "prob": f"{replace_prob:.2f}"})
        for idx in range(len(self.student.model.layers)):
            self.student.model.layers[idx] = original_layers[idx]
        self._cleanup_hooks()
        print("Calibration Completed!\n")