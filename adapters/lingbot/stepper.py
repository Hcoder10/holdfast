"""Step-per-chunk driver for LingBot-World 2 (causal-fast).

The stock pipeline takes a whole camera path up front and returns one video.
PlayWorld needs the opposite: generate a little, show the frames to the player
agent, get the next action, continue. This class keeps the model's caches alive
between calls so each `step()` adds one chunk (3 latent frames = 12 video frames
= 0.75 s at 16 fps), conditioned on camera motion and text chosen at that moment.

It mirrors `WanI2VCausal._generate_causal_fast` (lingbot-world-v2 @ 1895d30).
Differences from stock, all deliberate:
  * camera poses are supplied per chunk as frame-to-frame relative poses;
  * the text prompt may change between chunks (cross-attention cache is rebuilt);
  * noise is drawn per chunk, so seeds do not reproduce stock videos;
  * the image-condition latents past the first `COND_EXACT_LATENTS` reuse the
    last exact one (all-zero input frames), which lets rollouts run unbounded.
"""
import math
import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange

CHUNK = 3                 # latent frames per step
FRAMES_PER_CHUNK = 12     # video frames per step (first step yields 9)
FPS = 16
COND_EXACT_LATENTS = 21   # 81 video frames encoded exactly for the image condition
DECODE_CONTEXT_CHUNKS = 2
FAST_TIMESTEP_INDEX = [0, 179, 358, 679]


def rel_pose(yaw_deg=0.0, pitch_deg=0.0, forward=0.0, right=0.0, up=0.0):
    """Frame-to-frame camera motion as a 4x4 matrix in OpenCV camera coordinates
    (x right, y down, z forward). Positive yaw turns right, positive pitch looks
    up. Translation is in the model's normalised units: 1.0 is full speed."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx
    m[:3, 3] = [right, -up, forward]
    return m


class LingBotStepper:
    def __init__(self, repo, ckpt_dir, task="i2v-A14B", assets_dir=None,
                 max_area=480 * 832, local_attn_size=18, sink_size=6, device_id=0,
                 intrinsics=(503.0, 503.0, 416.0, 240.0)):
        repo = os.path.abspath(repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import wan
        import wan.image2video  # noqa: F401
        import wan.modules.model_fast  # noqa: F401
        from wan.configs import WAN_CONFIGS
        from wan.utils import cam_utils
        import sdpa_shim
        self.used_sdpa_shim = sdpa_shim.install()
        self.cam = cam_utils
        self.cfg = WAN_CONFIGS[task]
        self.max_area = max_area
        self.intrinsics_480 = intrinsics  # fx, fy, cx, cy for a 480x832 frame
        assert local_attn_size > 0, "streaming needs a bounded attention window"
        self.pipe = wan.WanI2VCausal(
            config=self.cfg, checkpoint_dir=ckpt_dir, device_id=device_id, rank=0,
            t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False,
            local_attn_size=local_attn_size, sink_size=sink_size,
            infer_mode="causal_fast", assets_dir=assets_dir)
        self.device = self.pipe.device
        self.pipe.text_encoder.model.to(self.device)
        self.chunk_id = None

    # ------------------------------------------------------------------ text
    def _encode(self, prompt):
        cache = self.pipe._t5_cache
        if prompt not in cache:
            cache[prompt] = self.pipe.text_encoder([prompt], self.device)
        return cache[prompt]

    def set_prompt(self, prompt):
        """Switch the text condition. Takes effect from the next step."""
        if prompt == getattr(self, "prompt", None):
            return
        p, m = self.pipe, self.pipe.model.config
        self.prompt = prompt
        self.context = self._encode(prompt)
        self.cross_cache = p._initialize_crossattn_cache(
            num_layers=m.num_layers,
            shape=[1, 512, m.num_heads, m.dim // m.num_heads],
            dtype=p.pipe_dtype, device=self.device)
        self.cross_first = True

    # ----------------------------------------------------------------- reset
    @torch.no_grad()
    def reset(self, image, prompt, seed=0):
        """Start an episode from a PIL image and a text prompt."""
        p = self.pipe
        vs, ps = p.vae_stride, p.patch_size
        img = TF.to_tensor(image.convert("RGB")).sub_(0.5).div_(0.5).to(self.device)
        h0, w0 = img.shape[1:]
        ar = h0 / w0
        self.lat_h = round(np.sqrt(self.max_area * ar) // vs[1] // ps[1] * ps[1])
        self.lat_w = round(np.sqrt(self.max_area / ar) // vs[2] // ps[2] * ps[2])
        self.h, self.w = self.lat_h * vs[1], self.lat_w * vs[2]
        self.frame_seqlen = self.lat_h * self.lat_w // (ps[1] * ps[2])
        self.max_seq_len = CHUNK * self.frame_seqlen
        self.kv_size = self.frame_seqlen * p.local_attn_size

        # image condition: mask (4 ch) + VAE latents of [image, zeros...] (16 ch)
        n_exact = (COND_EXACT_LATENTS - 1) * 4 + 1
        first = torch.nn.functional.interpolate(
            img[None].cpu(), size=(self.h, self.w), mode="bicubic").transpose(0, 1)
        video = torch.concat([first, torch.zeros(3, n_exact - 1, self.h, self.w)], dim=1)
        self.cond_lat = p.vae.encode([video.to(self.device)])[0]  # [16, 21, h, w]
        del video

        fx, fy, cx, cy = self.intrinsics_480
        ks = torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32)
        ks = self.cam.get_Ks_transformed(
            ks, height_org=480, width_org=832, height_resize=self.h,
            width_resize=self.w, height_final=self.h, width_final=self.w)
        self.ks = ks.repeat(CHUNK, 1).to(self.device)

        p.scheduler.set_timesteps(p.num_train_timesteps, shift=self.cfg.sample_shift)
        self.timesteps = p.scheduler.timesteps[FAST_TIMESTEP_INDEX]

        m = p.model.config
        self.self_cache = p._initialize_self_kv_cache(
            num_layers=m.num_layers,
            shape=[1, self.kv_size, m.num_heads, m.dim // m.num_heads],
            dtype=p.pipe_dtype, device=self.device)
        self.prompt = None
        self.set_prompt(prompt)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.latents = []       # one [16, 3, h, w] tensor per chunk
        self.rel_poses = []     # one [3, 4, 4] array per chunk, as supplied
        self.prompts = []       # prompt used for each chunk
        self.chunk_id = 0

    def _condition(self, chunk_id):
        idx = [min(chunk_id * CHUNK + i, COND_EXACT_LATENTS - 1) for i in range(CHUNK)]
        lat = self.cond_lat[:, idx]
        msk = torch.zeros(4, CHUNK, self.lat_h, self.lat_w, device=self.device)
        if chunk_id == 0:
            msk[:, 0] = 1
        return torch.concat([msk, lat])

    def _plucker(self, rel):
        c2w = torch.as_tensor(np.asarray(rel), dtype=torch.float32, device=self.device)
        emb = self.cam.get_plucker_embeddings(c2w, self.ks, self.h, self.w)
        emb = rearrange(emb, "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                        c1=self.h // self.lat_h, c2=self.w // self.lat_w)[None]
        emb = rearrange(emb, "b (f h w) c -> b c f h w",
                        f=CHUNK, h=self.lat_h, w=self.lat_w)
        return emb.to(self.pipe.param_dtype)

    # ----------------------------------------------------------------- guide
    @torch.no_grad()
    def _encode_guide(self, frames, mask, erode=1, feather=2):
        """Turn 12 remembered target frames (+ validity mask) into chunk latents.

        The VAE is causal in time, so the target frames are encoded after the
        last 9 frames actually shown, and the final three latents are kept.
        Returns (latents [16,3,h,w], mask [1,3,h,w] float) or None if too little
        of the chunk is covered."""
        ctx = self.shown[-9:]
        fill = frames.copy()
        fill[~mask] = np.broadcast_to(ctx[-1], frames.shape)[~mask]
        clip = np.concatenate([ctx, fill])  # 21 frames -> 6 latents
        v = torch.from_numpy(clip).to(self.device).permute(3, 0, 1, 2).float() / 127.5 - 1
        lat = self.pipe.vae.encode([v])[0][:, -CHUNK:]
        m = torch.from_numpy(mask).to(self.device).float()            # [12, H, W]
        m = m.view(CHUNK, 4, self.h, self.w).amin(dim=1)               # all 4 frames valid
        m = -torch.nn.functional.max_pool2d(-m[None], 8)[0]            # all 8x8 pixels valid
        for _ in range(erode):                                         # stay off the seam
            m = -torch.nn.functional.max_pool2d(-m[None], 3, stride=1, padding=1)[0]
        if float(m.mean()) < 0.05:
            return None
        if feather:  # ramp the edge so the model blends remembered and new content
            soft = m[None]
            for _ in range(feather):
                soft = torch.nn.functional.avg_pool2d(soft, 3, stride=1, padding=1)
            m = torch.minimum(m, soft[0] * soft[0])
        return lat.float(), m[None]

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, rel, prompt=None, decode=True, guide=None, guide_from=1, guide_hard=True):
        """Generate one chunk.

        rel: three 4x4 frame-to-frame relative camera poses (see `rel_pose`).
             On the first chunk the first pose is forced to identity.
        prompt: optional new text condition from this chunk on.
        guide: optional (frames uint8 [12,H,W,3], mask bool [12,H,W]) of remembered
             content for this chunk. Where the mask is set, the denoiser is held to
             the remembered latents from sampling step `guide_from` on, so the model
             redraws what was there and only invents the rest. With guide_hard the
             remembered latents also replace the final prediction in that region.
        Returns uint8 frames [T, H, W, 3] (T=9 on the first chunk, else 12), or
        None when decode=False.
        """
        assert self.chunk_id is not None, "call reset() first"
        p = self.pipe
        if prompt is not None:
            self.set_prompt(prompt)
        g = None
        if guide is not None and self.chunk_id > 0:
            g = self._encode_guide(*guide)
        self.last_guide_coverage = 0.0 if g is None else float(g[1].mean())
        rel = np.asarray(rel, dtype=np.float64).copy()
        assert rel.shape == (CHUNK, 4, 4)
        if self.chunk_id == 0:
            rel[0] = np.eye(4)
        kwargs = {
            "context": [self.context[0]],
            "seq_len": self.max_seq_len,
            "y": [self._condition(self.chunk_id)],
            "dit_cond_dict": {"c2ws_plucker_emb": self._plucker(rel).chunk(1, dim=0)},
            "kv_cache": self.self_cache,
            "crossattn_cache": self.cross_cache,
            "current_start": self.chunk_id * CHUNK * self.frame_seqlen,
            "max_attention_size": self.kv_size,
            "frame_seqlen": self.frame_seqlen,
        }
        latent = torch.randn(16, CHUNK, self.lat_h, self.lat_w, dtype=torch.float32,
                             generator=self.gen, device=self.device)
        with torch.amp.autocast("cuda", dtype=p.param_dtype):
            for i, t in enumerate(self.timesteps):
                if g is not None and i >= guide_from:
                    known = p.scheduler.add_noise(
                        g[0], torch.randn(g[0].shape, generator=self.gen, device=self.device,
                                          dtype=g[0].dtype), t)
                    latent = g[1] * known + (1 - g[1]) * latent
                pred = p.model(x=[latent], t=torch.stack([t]).to(self.device),
                               cross_attn_first_call=self.cross_first, **kwargs)[0]
                self.cross_first = False
                x0 = p._convert_flow_pred_to_x0(
                    flow_pred=pred, xt=latent, timestep=t, scheduler=p.scheduler)
                if i < len(self.timesteps) - 1:
                    noise = torch.randn(x0.shape, generator=self.gen,
                                        device=x0.device, dtype=x0.dtype)
                    latent = p.scheduler.add_noise(x0, noise, self.timesteps[i + 1])
            if g is not None and guide_hard:
                x0 = (g[1] * g[0] + (1 - g[1]) * x0).to(x0.dtype)
            # write the clean chunk into the attention cache
            p.model(x=[x0], t=torch.stack([self.timesteps[-1] * 0.0]).to(self.device),
                    cross_attn_first_call=False, **kwargs)
        self.latents.append(x0.float())
        self.rel_poses.append(rel)
        self.prompts.append(self.prompt)
        self.chunk_id += 1
        if not decode:
            return None
        frames = self._decode_tail()
        self.shown = frames if self.chunk_id == 1 else np.concatenate([self.shown[-12:], frames])
        return frames

    # ---------------------------------------------------------------- decode
    @staticmethod
    def _to_uint8(video):
        v = ((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return v.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, 3]

    @torch.no_grad()
    def _decode_tail(self):
        """Decode the newest chunk with a little left context for the causal VAE."""
        ctx = self.latents[-(DECODE_CONTEXT_CHUNKS + 1):]
        frames = self._to_uint8(self.pipe.vae.decode([torch.cat(ctx, dim=1)])[0])
        if len(self.latents) == 1:
            return frames
        return frames[-FRAMES_PER_CHUNK:]

    @torch.no_grad()
    def video(self):
        """Decode the whole episode in one pass (the authoritative video)."""
        return self._to_uint8(self.pipe.vae.decode([torch.cat(self.latents, dim=1)])[0])
