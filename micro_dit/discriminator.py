import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorCrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.eps = eps

        self.query_token = nn.Parameter(torch.randn(1, 1, dim) / math.sqrt(dim))

        self.norm = nn.LayerNorm(dim, eps=self.eps)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim, eps=eps)
            self.k_norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # query: [B,1,C]
        query = self.query_token.expand(B, -1, -1)

        x_norm = self.norm(x)

        q = self.q_proj(query)  # [B,1,dim]
        k = self.k_proj(x_norm) # [B,L,dim]
        v = self.v_proj(x_norm) # [B,L,dim]

        # optionally layernorm q/k
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        head_dim = self.dim // self.num_heads
        # [B,1,dim] -> [B, num_heads, 1, head_dim]
        q = q.reshape(B, 1, self.num_heads, head_dim).transpose(1, 2)
        # [B,L,dim] -> [B, num_heads, L, head_dim]
        k = k.reshape(B, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, head_dim).transpose(1, 2)

        # attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)  # [B,num_heads,1,L]

        out = torch.matmul(attn_weights, v) # [B,num_heads,1,head_dim]
        out = out.transpose(1,2).contiguous().view(B, 1, self.dim)
        out = self.o_proj(out)  # linear proj -> [B,1,dim]
        return out


class APTDiscriminator(nn.Module):
    def __init__(self, dit_model):
        super().__init__()

        self.backbone = copy.deepcopy(dit_model)
        self.dim = self.backbone.dit.dim
        self.num_heads = self.backbone.dit.head_dim

        self.qk_norm = True
        self.eps = 1e-6

        self.cross_attn_15 = DiscriminatorCrossAttnBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            qk_norm=self.qk_norm,
            eps=self.eps
        )
        self.cross_attn_25 = DiscriminatorCrossAttnBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            qk_norm=self.qk_norm,
            eps=self.eps
        )
        self.cross_attn_27 = DiscriminatorCrossAttnBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            qk_norm=self.qk_norm,
            eps=self.eps
        )
        self.final_proj = nn.Sequential(
            nn.LayerNorm(self.dim * 3, eps=self.eps),
            nn.Linear(self.dim * 3, 1)
        )

    def forward(self, noise, conditioning, t, return_features=False):
        layer_outputs = {}

        def hook_fn_15(module, inp, out):
            layer_outputs[15] = out
        def hook_fn_25(module, inp, out):
            layer_outputs[25] = out
        def hook_fn_27(module, inp, out):
            layer_outputs[27] = out

        h1 = self.backbone.dit.blocks[15].register_forward_hook(hook_fn_15)
        h2 = self.backbone.dit.blocks[25].register_forward_hook(hook_fn_25)
        h3 = self.backbone.dit.blocks[27].register_forward_hook(hook_fn_27)

        with torch.no_grad():
            self.backbone.dit.forward(
                x=noise,
                t=t,
                y=conditioning,
                mask_ratio=0,
                cfg=7.5
            )
        h1.remove()
        h2.remove()
        h3.remove()

        feat_15 = layer_outputs[15]  # [B,T,D]
        feat_25 = layer_outputs[25]  # [B,T,D]
        feat_27 = layer_outputs[27]  # [B,T,D]

        feat_15_out = self.cross_attn_15(feat_15)
        feat_25_out = self.cross_attn_25(feat_25)
        feat_27_out = self.cross_attn_27(feat_27)

        concat_feats = torch.cat([
            feat_15_out.squeeze(1),
            feat_25_out.squeeze(1),
            feat_27_out.squeeze(1)
        ], dim=-1)  # [B, 3*dim]

        logit = self.final_proj(concat_feats)  # [B,1]

        if return_features:
            return logit, (feat_15_out, feat_25_out, feat_27_out)
        return logit

def approximated_r1_loss(discriminator, real_image, timestep, conditioning, sigma=0.01):
    # Get discriminator prediction on real samples
    real_pred = discriminator(real_image, conditioning, timestep)
    # Add small Gaussian perturbation to real samples
    perturbed_samples = real_image + torch.randn_like(real_image) * sigma

    perturbed_pred = discriminator(perturbed_samples, conditioning, timestep)
    loss = torch.mean((real_pred - perturbed_pred) ** 2)
    return loss

class EMA:
    def __init__(self, model, decay=0.9995):
        self.model = copy.deepcopy(model)
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
