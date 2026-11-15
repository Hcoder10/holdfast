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
        self.handles = [model.blocks[i].register_forward_hook(self._hook(i)) for i in self.layers]

    def _hook(self, i):
        def fn(module, inputs, out):
            v = self.steer.get(i)
            if v is not None:
                out = out + v.to(device=out.device, dtype=out.dtype)
            if self.record:
                self.buf[i] = out.detach().float().mean(dim=1)[0].cpu()
            return out
        return fn

    def take(self):
        """The recorded vectors as one [n_layers, C] tensor, and clear the buffer."""
        out = torch.stack([self.buf[i] for i in self.layers]) if len(self.buf) == len(self.layers) else None
        self.buf = {}
        return out

    def set_steer(self, vectors, alpha=1.0):
        """vectors: {layer: 1-D array}; alpha scales all of them. Empty dict clears."""
        self.steer = {int(k): torch.as_tensor(v, dtype=torch.float32) * float(alpha) for k, v in vectors.items()}

    def remove(self):
        for h in self.handles:
            h.remove()
