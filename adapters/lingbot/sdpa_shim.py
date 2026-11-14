"""Drop-in replacement for LingBot-World's `flash_attention` using PyTorch SDPA.

The upstream cross-attention path calls `flash_attention` directly and asserts
that the flash-attn package is installed. flash-attn has no prebuilt kernels for
every GPU generation, so `install()` swaps in this version when the package is
missing. Key padding (`k_lens`) is honoured with a boolean mask.
"""
import torch
import torch.nn.functional as F

# World-anchored cross-attention. When set to a bool tensor [Lq, Lk] (image tokens
# x text tokens, True = may attend), cross-attention calls of exactly that shape use
# it. The stepper sets it for one generation step and clears it afterwards.
CROSS_MASK = None


def sdpa_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0,
                         softmax_scale=None, q_scale=None, causal=False,
                         window_size=(-1, -1), deterministic=False,
                         dtype=torch.bfloat16, version=None):
    # q: [B, Lq, N, C], k/v: [B, Lk, N, C]
    assert softmax_scale is None and window_size == (-1, -1)
    out_dtype = q.dtype
    half = (torch.float16, torch.bfloat16)
    if q.dtype not in half:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if q_scale is not None:
        q = q * q_scale
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    mask = None
    if k_lens is not None:
        idx = torch.arange(lk, device=q.device)[None, :]
        mask = (idx < k_lens.to(q.device)[:, None])[:, None, None, :]  # [B,1,1,Lk]
    cm = CROSS_MASK
    if cm is not None and cm.shape == (lq, lk):
        cm = cm.to(q.device)[None, None]
        mask = cm if mask is None else (mask & cm)
    x = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, is_causal=causal, dropout_p=dropout_p)
    x = x.transpose(1, 2).contiguous()
    if q_lens is not None:
        idx = torch.arange(lq, device=q.device)[None, :]
        x = x * (idx < q_lens.to(q.device)[:, None])[:, :, None, None]
    return x.type(out_dtype)


def install(force=False):
    """Patch every loaded `wan` module that imported `flash_attention`.
    Returns True if the shim was installed, False if flash-attn is present and
    `force` is not set. World-anchored cross-attention needs the shim."""
    import sys
    import wan.modules.attention as att
    if (att.FLASH_ATTN_2_AVAILABLE or att.FLASH_ATTN_3_AVAILABLE) and not force:
        return False
    att.flash_attention = sdpa_flash_attention
    for name, mod in list(sys.modules.items()):
        if name.startswith("wan.") and getattr(mod, "flash_attention", None) is not None:
            mod.flash_attention = sdpa_flash_attention
    return True
