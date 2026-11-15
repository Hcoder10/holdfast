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

# Cross-attention register gain. When set to a float tensor [Lk] (one value per text
# token), it is added to the attention logits of every cross-attention call with that
# many text tokens: log(w) on a span makes image tokens attend to it w times as much.
CROSS_BIAS = None
BIAS_HITS = 0      # calls in which the bias was applied; a silent no-op would leave this at 0


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
    cb = CROSS_BIAS
    if cb is not None and cb.shape[0] == lk:
        global BIAS_HITS
        BIAS_HITS += 1
        cb = cb.to(device=q.device, dtype=q.dtype)[None, None, None, :]
        mask = cb if mask is None else torch.where(mask, cb, torch.full_like(cb, float("-inf")))
    x = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, is_causal=causal, dropout_p=dropout_p)
    x = x.transpose(1, 2).contiguous()
    if q_lens is not None:
        idx = torch.arange(lq, device=q.device)[None, :]
        x = x * (idx < q_lens.to(q.device)[:, None])[:, :, None, None]
    return x.type(out_dtype)


# Self-attention gain on the sink (the episode's first frames, kept at the front of the
# attention cache). (n_sink_tokens, log_gain): added to the logits of those keys in
# self-attention calls whose cache already extends past the sink and the current chunk.
SELF_SINK = None
SELF_HITS = 0
# Diagnostic: share of self-attention that lands on the sink keys, measured directly
# (explicit softmax over a sample of queries) without and with the gain. Filled only
# while SELF_DIAG_ON is set; one entry per measured call.
SELF_DIAG_ON = False
SELF_DIAG = []


def self_attention(q, k, v, **kw):
    """Stand-in for wan.modules.attention.attention as used by the cached self-attention
    (its SDPA branch), plus the optional sink gain."""
    global SELF_HITS
    dtype = kw.get("dtype", torch.bfloat16)
    lq, lk = q.size(1), k.size(1)
    mask = None
    ss = SELF_SINK
    if ss is not None and lk >= ss[0] + 2 * lq:
        mask = torch.zeros(lk, device=q.device, dtype=dtype)
        mask[:ss[0]] = ss[1]
        mask = mask[None, None, None, :]
        SELF_HITS += 1
        if SELF_DIAG_ON and SELF_HITS % 97 == 1 and len(SELF_DIAG) < 400:
            qs = q[0, ::max(1, lq // 48)].float().transpose(0, 1)          # [heads, m, C]
            ks = k[0].float().transpose(0, 1)                               # [heads, lk, C]
            logits = torch.einsum("hmc,hkc->hmk", qs, ks) / (q.size(-1) ** 0.5)
            plain = logits.softmax(-1)[..., :ss[0]].sum(-1).mean().item()
            biased = (logits + mask[0, 0].float()).softmax(-1)[..., :ss[0]].sum(-1).mean().item()
            top = logits.softmax(-1).max(-1).values.mean().item()
            SELF_DIAG.append({"hit": SELF_HITS, "lk": lk, "sink_share_plain": round(plain, 4),
                              "sink_share_with_gain": round(biased, 4), "mean_top1_weight": round(top, 4)})
    out = F.scaled_dot_product_attention(q.transpose(1, 2).to(dtype), k.transpose(1, 2).to(dtype),
                                         v.transpose(1, 2).to(dtype), attn_mask=mask)
    return out.transpose(1, 2).contiguous()


def install(force=False):
    """Patch every loaded `wan` module that imported `flash_attention`.
    Returns True if the shim was installed, False if flash-attn is present and
    `force` is not set. World-anchored cross-attention needs the shim."""
    import sys
    import wan.modules.attention as att
    # The cached self-attention calls `attention` (SDPA when flash-attn is absent). Route it
    # here in every model module that imported it, so the sink gain can apply.
    import importlib
    for name in ("wan.modules.model_causal", "wan.modules.model_fast"):
        try:
            mod = importlib.import_module(name)
            if getattr(mod, "attention", None) is not None:
                mod.attention = self_attention
        except Exception:
            pass
    if (att.FLASH_ATTN_2_AVAILABLE or att.FLASH_ATTN_3_AVAILABLE) and not force:
        return False
    att.flash_attention = sdpa_flash_attention
    for name, mod in list(sys.modules.items()):
        if name.startswith("wan.") and getattr(mod, "flash_attention", None) is not None:
            mod.flash_attention = sdpa_flash_attention
    return True
