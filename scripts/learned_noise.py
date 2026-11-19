"""
Learned noise-aware attention for Retinexformer (second contribution attempt).

Why this exists
---------------
The parameter-free SNR gate (`snr_guidance.py`) failed: control beat guided by 0.38 dB.
Diagnosis: a fixed multiplier on attention values adds no capacity to *decide* how to handle
noise, and the hand-made |input - blur| map also fires on edges.

This module makes the noise signal learned:
  1. `NoiseEstimator`: a small CNN that maps the low-light input to a per-pixel noise score.
  2. In every IG_MSA, a learned 1x1 conv fuses the noise score with the illumination feature
     before it multiplies the value features: v = v * illu * (1 + gate(noise)).

Crucial checkpoint property
---------------------------
The estimator and fusion parameters are NEW keys, absent from the released checkpoint.
All original Retinexformer keys keep their exact names/shapes, so
`model.load_state_dict(pretrained, strict=False)` loads the released weights perfectly and
only the new modules are randomly initialised. With `--noise-scale 0`, the new modules are
not used at all and the forward pass is bit-identical to official Retinexformer.

Usage (training, from scratch OR pretrained-initialised):
    python scripts/finetune.py --loss l1 --iters 150000 --lr 2e-4 --seed 42 \
        --pretrained pretrained_weights/LOL_v2_synthetic.pth \
        --learned-noise --noise-scale 0.5 --opt <yml> --data <train> --out <dir>

Usage (inference):
    python scripts/retinexformer_infer.py --weights <pth> --learned-noise --noise-scale 0.5 ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .snr_guidance import _guided_attention_forward  # noqa: F401 (keep import surface)
except ImportError:
    pass


class NoiseEstimator(nn.Module):
    """Predict a per-pixel noise score in [0, 1] from the low-light RGB input."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, low):
        return self.net(low)  # (B, 1, H, W) in [0, 1]


def _noise_attention_forward(self, x_in, illu_fea_trans):
    """IG-MSA forward with learned noise modulation of the value features."""
    if self._noise_gate is None:
        raise RuntimeError("Noise guidance enabled but no input noise gate was set")

    from einops import rearrange

    b, h, w, c = x_in.shape
    x = x_in.reshape(b, h * w, c)
    q_inp = self.to_q(x)
    k_inp = self.to_k(x)
    v_inp = self.to_v(x)
    q, k, v, illu_attn = map(
        lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
        (q_inp, k_inp, v_inp, illu_fea_trans.flatten(1, 2)),
    )

    # Resize the input-resolution noise map to this feature level and learn its contribution.
    noise = F.interpolate(self._noise_gate, size=(h, w), mode="bilinear", align_corners=False)
    noise = noise.flatten(2).transpose(1, 2)                       # (B, HW, 1)
    illu_flat = illu_fea_trans.flatten(1, 2)                       # (B, HW, C): input is [b,h,w,c]
    # The fusion conv expects channels-last [B, HW, C+1] via a linear layer on the concatenated map.
    fused = self.noise_fuse(torch.cat([illu_flat, noise], dim=-1))  # (B, HW, C)
    fused = rearrange(fused, "b n (h d) -> b h n d", h=self.num_heads)  # (B, heads, HW, d)

    # v is already head-split above; scale it by the learned illumination+noise gate.
    v = v * fused

    q = q.transpose(-2, -1)
    k = k.transpose(-2, -1)
    v = v.transpose(-2, -1)
    q = F.normalize(q, dim=-1, p=2)
    k = F.normalize(k, dim=-1, p=2)
    attn = (k @ q.transpose(-2, -1)) * self.rescale
    attn = attn.softmax(dim=-1)
    x = attn @ v
    x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
    out_c = self.proj(x).view(b, h, w, c)
    out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return out_c + out_p


def enable_learned_noise(model):
    """Attach a noise estimator and per-block fusion linears; return the number of guided blocks."""
    if getattr(model, "_learned_noise_enabled", False):
        raise RuntimeError("Learned noise guidance is already enabled on this model")

    # One estimator at the front of each stage: the whole model input is the low-light image.
    estimators = nn.ModuleList()

    def replace_blocks(parent):
        converted = 0
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ == "IG_MSA":
                base_class = child.__class__

                def init_from_base(self, base):
                    base_class.__init__(self, dim=base.dim, dim_head=base.dim_head, heads=base.num_heads)
                    self.load_state_dict(base.state_dict(), strict=True)  # original keys intact
                    self.noise_fuse = nn.Linear(base.dim + 1, base.dim, bias=True)
                    # Initialise fusion as a pass-through of the illumination feature: W = [I, 0], b = 0.
                    with torch.no_grad():
                        self.noise_fuse.weight.zero_()
                        self.noise_fuse.weight[:, :base.dim].copy_(torch.eye(base.dim))
                        # Small nonzero noise column so gradients reach the estimator from step 1.
                        nn.init.normal_(self.noise_fuse.weight[:, base.dim:], mean=0, std=1e-3)
                        self.noise_fuse.bias.zero_()
                    self._noise_gate = None

                guided_class = type(
                    "NoiseGuidedIG_MSA",
                    (base_class,),
                    {
                        "__init__": init_from_base,
                        "forward": _noise_attention_forward,
                        "_uses_noise_gate": True,
                    },
                )
                setattr(parent, name, guided_class(child))
                converted += 1
            else:
                converted += replace_blocks(child)
        return converted

    n_blocks = replace_blocks(model)
    if n_blocks == 0:
        raise ValueError("No IG_MSA blocks found")

    # Estimator is attached to the model. During guided_forward it is fed the ORIGINAL low-light image.
    model.estimator_noise = NoiseEstimator()
    model._learned_noise_enabled = True
    model._noise_scale = 0.0
    return n_blocks


def guided_forward(model, low, scale):
    """Run the model with a learned noise map. scale=0 -> identical to the original model."""
    blocks = [m for m in model.modules() if getattr(m, "_uses_noise_gate", False)]
    if not blocks:
        raise RuntimeError("Learned noise guidance requested on a model without guided blocks")
    if scale < 0:
        raise ValueError("noise scale must be non-negative")

    if scale == 0:
        for block in blocks:
            block._noise_gate = torch.zeros_like(low[:, :1])
        try:
            return model(low)
        finally:
            for block in blocks:
                block._noise_gate = None

    noise_map = model.estimator_noise(low)              # (B,1,H,W), learned
    # Keep the initial behaviour close to the original: only perturb the gate by `scale * noise`.
    gate = 1.0 + scale * (noise_map - 0.5)
    for block in blocks:
        block._noise_gate = gate
    try:
        return model(low)
    finally:
        for block in blocks:
            block._noise_gate = None
