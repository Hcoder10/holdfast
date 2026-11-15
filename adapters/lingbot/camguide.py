"""Camera guidance in the hidden state.

Every block of the model applies the camera as a scale and shift of its hidden state:
x <- (1 + scale(cam)) * x + shift(cam). On many real-photo scenes the model under-responds
to a commanded move (the judge: "the camera stayed completely still"). Raising the input
does not help: translation is normalised to a maximum of 1.0 in training, and beyond that
the picture breaks.

Here the *effect* of the command is amplified instead. For each block, the scale and shift
a still camera would produce are computed once; the commanded pose's scale and shift are
then extrapolated away from them:

    scale_w = scale_still + w * (scale_cmd - scale_still)      (same for shift)

w = 1 is the unmodified model. A still chunk is unchanged at any w. This is the camera
analogue of classifier-free guidance, done inside the forward pass at the cost of two
small linear layers per block.
"""
import torch
import torch.nn.functional as F
from einops import rearrange


class CamGuide:
    def __init__(self, model):
        self.m, self.w, self.still, self.busy, self.calls = model, 1.0, None, False, 0
        for i, blk in enumerate(model.blocks):
            blk.cam_scale_layer.register_forward_hook(self._hook(i, 0))
            blk.cam_shift_layer.register_forward_hook(self._hook(i, 1))

    @torch.no_grad()
    def prepare(self, plucker_still, dtype):
        """plucker_still: [1, c, f, h, w] for a camera that does not move."""
        m, ps = self.m, self.m.patch_size
        self.busy = True
        try:
            with torch.amp.autocast("cuda", dtype=dtype):
                e = rearrange(plucker_still, "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                              c1=ps[0], c2=ps[1], c3=ps[2])
                e = m.patch_embedding_wancamctrl(e)
                e = e + m.c2ws_hidden_states_layer2(F.silu(m.c2ws_hidden_states_layer1(e)))
                self.still = []
                for blk in m.blocks:
                    h = blk.cam_injector_layer2(F.silu(blk.cam_injector_layer1(e))) + e
                    self.still.append((blk.cam_scale_layer(h), blk.cam_shift_layer(h)))
        finally:
            self.busy = False

    def _hook(self, i, j):
        def fn(mod, inp, out):
            if self.busy or self.w == 1.0 or self.still is None:
                return out
            st = self.still[i][j]
            if st.shape != out.shape:
                return out
            self.calls += 1
            return st.to(out.dtype) + self.w * (out - st.to(out.dtype))
        return fn
