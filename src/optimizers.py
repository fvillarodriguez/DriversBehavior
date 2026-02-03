import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion optimizer (EvoLved Sign Momentum)."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group.get("weight_decay", 0.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)

                if weight_decay:
                    p.add_(p, alpha=-lr * weight_decay)
                p.add_(update.sign(), alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def get_optimizer_cls(name: str):
    if not name:
        return torch.optim.Adam
    key = str(name).lower()
    if key == "lion":
        return Lion
    return getattr(torch.optim, str(name), torch.optim.Adam)
