"""SNR-guided value modulation for Retinexformer's illumination-guided attention."""

import torch.nn.functional as F

try:
    from .losses import snr_weights
except ImportError:
    from losses import snr_weights


def _guided_attention_forward(self, x_in, illu_fea_trans):
    """Use a spatial SNR gate to scale attention values without adding parameters."""
    if self._snr_gate is None:
        raise RuntimeError("SNR guidance is enabled but no input SNR gate was set")

    from einops import rearrange

    b, h, w, c = x_in.shape
    x = x_in.reshape(b, h * w, c)
    q_inp = self.to_q(x)
    k_inp = self.to_k(x)
    v_inp = self.to_v(x)
    illu_attn = illu_fea_trans
    q, k, v, illu_attn = map(
        lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
        (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)),
    )

    # Interpolate the input-resolution gate for this encoder/decoder feature map.
    gate = F.interpolate(
        self._snr_gate, size=(h, w), mode="bilinear", align_corners=False
    )
    gate = gate.flatten(2).transpose(1, 2).unsqueeze(1).to(dtype=v.dtype)
    v = v * illu_attn * gate

    q = q.transpose(-2, -1)
    k = k.transpose(-2, -1)
    v = v.transpose(-2, -1)
    q = F.normalize(q, dim=-1, p=2)
    k = F.normalize(k, dim=-1, p=2)
    attn = (k @ q.transpose(-2, -1)) * self.rescale
    attn = attn.softmax(dim=-1)
    x = attn @ v
    x = x.permute(0, 3, 1, 2).reshape(
        b, h * w, self.num_heads * self.dim_head
    )
    out_c = self.proj(x).view(b, h, w, c)
    out_p = self.pos_emb(
        v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1)
    return out_c + out_p


def enable_snr_guidance(model):
    """Replace IG-MSA blocks with parameter-compatible SNR-guided blocks in place."""
    converted = 0

    def replace_blocks(parent):
        nonlocal converted
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ == "IG_MSA":
                base_class = child.__class__

                def init_from_base(self, base):
                    base_class.__init__(
                        self,
                        dim=base.dim,
                        dim_head=base.dim_head,
                        heads=base.num_heads,
                    )
                    self.load_state_dict(base.state_dict(), strict=True)
                    self._snr_gate = None

                guided_class = type(
                    "SNRGuidedIG_MSA",
                    (base_class,),
                    {
                        "__init__": init_from_base,
                        "forward": _guided_attention_forward,
                        "_uses_snr_gate": True,
                    },
                )
                setattr(parent, name, guided_class(child))
                converted += 1
            else:
                replace_blocks(child)

    replace_blocks(model)
    if converted == 0:
        raise ValueError("No Retinexformer IG_MSA blocks were found to guide")
    return converted


def guided_forward(model, low, alpha):
    """Run a model with a rank-normalized, mean-one SNR gate from its low-light input."""
    if alpha < 0:
        raise ValueError("SNR attention alpha must be non-negative")
    blocks = [module for module in model.modules() if getattr(module, "_uses_snr_gate", False)]
    if not blocks:
        raise RuntimeError("SNR guidance was requested on a model without guided IG_MSA blocks")

    gate = snr_weights(low, alpha=alpha)
    for block in blocks:
        block._snr_gate = gate
    try:
        return model(low)
    finally:
        for block in blocks:
            block._snr_gate = None
