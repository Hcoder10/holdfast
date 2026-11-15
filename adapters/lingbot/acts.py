"""Read and write the model's residual stream at chosen transformer blocks.

Reading: with `record` on, the mean over image tokens of each chosen block's output
is kept for the current forward pass. The stepper turns it on only for the clean
(timestep 0) pass that writes a finished chunk into the attention cache, so there is
one comparable activation vector per chunk and per layer.

Writing: `steer[layer]` is a vector added to that block's output on every forward
pass (denoising and cache write alike). This is how a direction found from judged
runs is programmed back into the model: no weights change, and the vector, its
source and its strength are all inspectable.
"""
import torch

LAYERS = (4, 9, 14, 19, 24, 29, 34, 39)


class Probe:
    def __init__(self, model, layers=LAYERS):
        self.layers = tuple(layers)
        self.record = False
        self.buf = {}
        self.steer = {}       # layer -> tensor [C]
        self.op, self.alpha = "add", 1.0
        self.gate = True      # the stepper closes this for passes that should not be written to
        self.writes = 0
        self.handles = [model.blocks[i].register_forward_hook(self._hook(i)) for i in self.layers]

    def _hook(self, i):
        def fn(module, inputs, out):
            v = self.steer.get(i) if self.gate else None
            if v is not None:
                self.writes += 1
                v = v.to(device=out.device, dtype=out.dtype)
                if self.op == "ablate":
                    # directional ablation: remove (a fraction of) the component along v
                    u = v / (v.norm() + 1e-6)
                    out = out - self.alpha * (out @ u).unsqueeze(-1) * u
                else:
                    out = out + v
            if self.record:
                self.buf[i] = out.detach().float().mean(dim=1)[0].cpu()
            return out
        return fn

    def take(self):
        """The recorded vectors as one [n_layers, C] tensor, and clear the buffer."""
        out = torch.stack([self.buf[i] for i in self.layers]) if len(self.buf) == len(self.layers) else None
        self.buf = {}
        return out

    def set_steer(self, vectors, alpha=1.0, op="add", layers=None):
        """vectors: {layer: 1-D array}. op "add": add alpha * vector to the block output.
        op "ablate": remove alpha of the output's component along the vector (alpha 1 =
        full directional ablation). `layers` restricts to some blocks. Empty dict clears."""
        self.op, self.alpha = op, float(alpha)
        scale = float(alpha) if op == "add" else 1.0
        self.steer = {int(k): torch.as_tensor(v, dtype=torch.float32) * scale for k, v in vectors.items()
                      if layers is None or int(k) in layers}

    def remove(self):
        for h in self.handles:
            h.remove()
