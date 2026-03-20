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
        torch.manual_seed(42) 
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.teacher.eval()
        self.student.eval()
        candidates = []
        num_candidates = num_samples
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
                candidates.append(out.cpu())
        return candidates
    """        
    def calibrate(self, epochs=3, lr=1e-4, batch_size=1):
        calib_data = self.generate_synthetic_data(num_samples=256)
        trainable_params = [p for n, p in self.student.named_parameters() if "alphas" in n]
        original_layers = [layer for layer in self.teacher.model.layers]
        optimizer = torch.optim.Adam(trainable_params, lr=lr, eps=1e-6)
        self._setup_hooks()
        for epoch in range(epochs):
            self.student.train()
            total_loss = 0
            replace_prob = 0.5 * max(0, 1 - epoch / (epochs * 0.5))
            pbar = tqdm(enumerate(calib_data), total=len(calib_data), desc=f"Epoch {epoch+1}")
            for i, batch in pbar:
                s_output = self.student(batch)
                h_teacher = self.teacher.model.embed_tokens(batch)
                l_feat = 0
                for idx in range(len(self.teacher.model.layers)):
                    s_feat = self.s_hooks[idx].features
                    if torch.rand(1).item() < replace_prob:
                        with torch.no_grad():
                            res = self.student.model.layers[idx](h_teacher)
                            h_teacher = res[0] if isinstance(res, tuple) else res
                            self.t_hooks[idx].features = h_teacher
                            self.s_hooks[idx].features = s_feat
                    else:
                        with torch.no_grad():
                            res = self.teacher.model.layers[idx](h_teacher)
                            h_teacher = res[0] if isinstance(res, tuple) else res
                    t_feat = self.t_hooks[idx].features
                    if s_feat is not None and t_feat is not None:
                        l_feat += F.mse_loss(s_feat, t_feat)
                with torch.no_grad():
                    t_output = self.teacher.lm_head(self.teacher.model.norm(h_teacher))
                l_mse = F.mse_loss(s_output.logits, t_output.detach())
                loss = self.s1 * l_mse + self.s2 * l_feat
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "prob": f"{replace_prob:.2f}"})
                for h in self.s_hooks + self.t_hooks:
                    h.features = None
        self._cleanup_hooks()
        print("Calibration Completed!\n")
    """
    """
    def calibrate(self, epochs=3, lr=1e-4, batch_size=1):
        calib_data = self.generate_synthetic_data(num_samples=256)
        trainable_params = [p for n, p in self.student.named_parameters() if "alphas" in n]
        optimizer = torch.optim.Adam(trainable_params, lr=lr, eps=1e-6)
        self._setup_hooks()
        for epoch in range(epochs):
            self.student.train()
            total_loss = 0
            replace_prob = 0.5 * max(0, 1 - epoch / (epochs * 0.5))
            pbar = tqdm(enumerate(calib_data), total=len(calib_data), desc=f"Epoch {epoch+1}")
            for i, batch in pbar:
                batch = batch.to(self.device)
                batch_size, seq_len = batch.shape
                position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
                with torch.no_grad():
                    h_teacher = self.teacher.model.embed_tokens(batch)
                    causal_mask = self.teacher.model._update_causal_mask(
                        None, batch, h_teacher if 'h_teacher' in locals() else None, False
                    )
                    for idx in range(len(self.teacher.model.layers)):
                        layer_kwargs = {
                            "position_ids": position_ids,
                            "attention_mask": causal_mask,
                            "use_cache": False
                        }
                        if torch.rand(1).item() < replace_prob:
                            res = self.student.model.layers[idx](h_teacher, **layer_kwargs)
                            h_teacher = res[0] if isinstance(res, tuple) else res
                            self.t_hooks[idx].features = h_teacher
                        else:
                            res = self.teacher.model.layers[idx](h_teacher, **layer_kwargs)
                            h_teacher = res[0] if isinstance(res, tuple) else res
                    t_output = self.teacher.lm_head(self.teacher.model.norm(h_teacher))
                s_output = self.student(batch)
                l_feat = 0
                for idx in range(len(self.student.model.layers)):
                    s_feat = self.s_hooks[idx].features
                    t_feat = self.t_hooks[idx].features
                    if s_feat is not None and t_feat is not None:
                        l_feat += F.mse_loss(s_feat, t_feat)
                l_mse = F.mse_loss(s_output.logits, t_output.detach())
                loss = self.s1 * l_mse + self.s2 * l_feat
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "prob": f"{replace_prob:.2f}"})
                for h in self.s_hooks + self.t_hooks:
                    h.features = None
        self._cleanup_hooks()
        print("Calibration Completed!\n")
    """
    def calibrate(self, epochs=3, lr=1e-4, batch_size=1):
        torch.manual_seed(42) 
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        calib_data = self.generate_synthetic_data(num_samples=64, seq_len=64)
        trainable_params = [p for n, p in self.student.named_parameters() if "alphas" in n]
        optimizer = torch.optim.Adam(trainable_params, lr=lr, eps=1e-6)
        origin_teacher = [layer for layer in self.teacher.model.layers]
        self._setup_hooks()
        for epoch in range(epochs):
            self.student.train()
            total_loss = 0
            replace_prob = 0.5 * max(0, 1 - epoch / (epochs * 0.5))
            pbar = tqdm(enumerate(calib_data), total=len(calib_data), desc=f"Epoch {epoch+1}")
            for i, batch in pbar:
                batch = batch.to(self.device)
                with torch.no_grad():
                    for idx in range(len(self.teacher.model.layers)):
                        if torch.rand(1).item() < replace_prob:
                            self.teacher.model.layers[idx] = self.student.model.layers[idx]
                        else:
                            self.teacher.model.layers[idx] = origin_teacher[idx]
                    t_output = self.teacher(batch)
                    for idx in range(len(self.teacher.model.layers)):
                        if self.t_hooks[idx].features is None:
                            self.t_hooks[idx].features = self.s_hooks[idx].features
                            self.s_hooks[idx].features = None
                            self.teacher.model.layers[idx] = origin_teacher[idx]
                s_output = self.student(batch)
                l_feat = 0
                for idx in range(len(self.teacher.model.layers)):
                    t_feat = self.t_hooks[idx].features
                    s_feat = self.s_hooks[idx].features
                    if t_feat is not None and s_feat is not None:
                        l_feat += F.mse_loss(s_feat, t_feat)
                l_mse = F.mse_loss(s_output.logits, t_output.logits)
                loss = self.s1 * l_mse + self.s2 * l_feat
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "prob": f"{replace_prob:.2f}"})
                for h in self.s_hooks + self.t_hooks:
                    h.features = None
        self._cleanup_hooks()
        print("Calibration Completed!\n")