import torch

class QBB_v1:

    def __init__(self, k=4, learning_rate=1e-3, iteration=50):
        self.K = k
        self.lr = learning_rate
        self.iter = iteration
        self.q_min = -2
        self.q_max = 1
    """
    def _search_best_alpha(self, R):
        mean_val = R.abs().mean(dim=1, keepdim=True)
        base_alpha = mean_val / 3.0
        best_mse = torch.full((R.shape[0], 1), float('inf'), device=R.device)
        best_alpha = base_alpha.clone()
        for ratio in torch.linspace(0.5, 1.0, 20, device=R.device):
            test_alpha = base_alpha * ratio
            test_alpha = torch.clamp(test_alpha, min=1e-8)
            q = torch.clamp(torch.round(R / test_alpha), self.q_min, self.q_max)
            w_recon = q * test_alpha
            mse = torch.mean((w_recon - R)**2, dim=1, keepdim=True)
            mask = mse < best_mse
            best_mse[mask] = mse[mask]
            best_alpha[mask] = test_alpha[mask]
        return best_alpha
    """
    def _search_best_alpha(self, R):
        mean_val = R.abs().mean(dim=1, keepdim=True)
        base_alpha = mean_val / 0.8
        best_mse = torch.full((R.shape[0], 1), float('inf'), device=R.device)
        best_alpha = base_alpha.clone()
        for ratio in torch.linspace(0.5, 2.0, 40, device=R.device):
            test_alpha = base_alpha * ratio
            test_alpha = torch.clamp(test_alpha, min=1e-8)
            q = torch.clamp(torch.round(R / test_alpha), self.q_min, self.q_max)
            w_recon = q * test_alpha
            mse = torch.mean((w_recon - R)**2, dim=1, keepdim=True)
            mask = mse < best_mse
            best_mse[mask] = mse[mask]
            best_alpha[mask] = test_alpha[mask]
        return best_alpha
    """
    def decompose(self, W):
        R = W.clone().to(torch.float32)
        bases = []
        alphas = []
        errors = []
        errors.append(torch.mean(R**2))
        for i in range(self.K):
            B = torch.sign(R)
            B[B == 0] = 1
            B = B.to(torch.int8)
            alpha = torch.mean(torch.abs(R), dim=1, keepdim=True)
            R -= B * alpha
            error = torch.mean(R**2)
            errors.append(error)
            alphas.append(alpha)
            bases.append(B)
            print(f"Iter {i + 1}: error = {error:.8f}")
        return bases, alphas, errors
    """

    def decompose(self, W):
        Wf = W.to(torch.float32)
        R = Wf.clone()
        bases = []
        alphas = []
        errors = []
        self.q_min, self.q_max = -2, 1
        for i in range(self.K):
            alpha = self._search_best_alpha(R)
            B_float = torch.clamp(torch.round(R / (alpha + 1e-9)), self.q_min, self.q_max)
            B_int8 = B_float.to(torch.int8)
            R = R - B_int8.to(torch.float32) * alpha
            error = torch.mean(R**2).item()
            errors.append(error)
            alphas.append(alpha.to(W.dtype))
            bases.append(B_int8)
            print(f"  Layer {i+1}: MSE = {error:.8f}")
        return bases, alphas, errors
    
    def upd(self, W, bases, alphas, steps=3):
        device = W.device
        alphas_param = torch.nn.Parameter(torch.stack(alphas).to(device))
        bases_tensor = torch.stack([b.to(torch.float32) for b in bases]).to(device)
        optimizer = torch.optim.Adam([alphas_param], lr=self.lr, eps=1e-5)
        criterion = torch.nn.MSELoss()
        Wf = W.to(torch.float32)
        for j in range(steps):
            static_bases = bases_tensor.detach()
            for i in range(self.iter):
                W_hat = torch.sum(alphas_param * static_bases, dim=0)
                loss = criterion(W_hat, Wf)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_([alphas_param], clip_value=0.1)
                optimizer.step()
            with torch.no_grad():
                for k in range(self.K):
                    W_others = torch.sum(alphas_param * bases_tensor, dim=0) - (alphas_param[k] * bases_tensor[k])
                    R = Wf - W_others
                    nB = torch.clamp(torch.round(R / (alphas_param[k] + 1e-9)), self.q_min, self.q_max)
                    bases_tensor[k] = nB
            print(f"Outer Step {j + 1} | MSE loss: {loss.item():.8f}")
        return [b.to(torch.int8) for b in torch.unbind(bases_tensor, dim=0)], torch.unbind(alphas_param.detach(), dim=0)
    
    def reconstruct(self, bases, alphas):
        W = torch.zeros_like(bases[0], dtype=torch.float32)
        for B, alpha in zip(bases, alphas):
            W += B.to(torch.float32) * alpha
        return W