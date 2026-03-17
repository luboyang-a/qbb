import torch

class QBB_v1:

    def __init__(self, k=4, learning_rate=1e-3, iteration=50):
        self.K = k
        self.lr = learning_rate
        self.iter = iteration

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
                    nB= torch.sign(R)
                    nB[nB == 0] = 1
                    bases_tensor[k] = nB
            print(f"Outer Step {j + 1} | MSE loss: {loss.item():.8f}")
        return [b.to(torch.int8) for b in torch.unbind(bases_tensor, dim=0)], torch.unbind(alphas_param.detach(), dim=0)
    
    def reconstruct(self, bases, alphas):
        W = torch.zeros_like(bases[0], dtype=torch.float32)
        for B, alpha in zip(bases, alphas):
            W += B.to(torch.float32) * alpha
        return W