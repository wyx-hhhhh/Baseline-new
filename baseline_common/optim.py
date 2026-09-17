"""AdamW with FP32 CPU master parameters for the 24 GB GPU profile."""
import torch

class CPUAdamW:
    """Explicit synchronous offload; portable, slower than fused DeepSpeed Adam.

    GPU parameters and gradients retain the chosen model dtype; master weights
    and Adam moments are FP32 on CPU. Only optimizer steps transfer tensors.
    """
    def __init__(self, parameters, **kwargs):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.masters = [torch.nn.Parameter(p.detach().float().cpu().clone()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, **kwargs)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        for p in self.parameters:
            p.grad = None
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        for p, master in zip(self.parameters, self.masters):
            master.grad = None if p.grad is None else p.grad.detach().float().cpu()
        self.optimizer.step()
        for p, master in zip(self.parameters, self.masters):
            p.copy_(master.to(device=p.device, dtype=p.dtype))

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "masters": [p.detach() for p in self.masters]}

    def load_state_dict(self, state):
        if len(state["masters"]) != len(self.masters):
            raise ValueError("CPU optimizer parameter count changed")
        with torch.no_grad():
            for p, master, saved in zip(self.parameters, self.masters, state["masters"]):
                master.copy_(saved)
                p.copy_(master.to(device=p.device, dtype=p.dtype))
        self.optimizer.load_state_dict(state["optimizer"])
