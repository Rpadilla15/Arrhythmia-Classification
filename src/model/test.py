# Revised implementation with vectorized masking / unshuffling to avoid slow Python loops.
# Re-run the same sanity checks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class PatchEmbedding1D(nn.Module):
    def __init__(self, signal_length: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert signal_length % patch_size == 0, "signal_length must be divisible by patch_size"
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x

class RelativePositionBias(nn.Module):
    def __init__(self, max_len: int, num_heads: int):
        super().__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.bias_table = nn.Parameter(torch.zeros(2 * max_len - 1, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
    def forward(self, qlen: int, klen: int, device: Optional[torch.device] = None) -> torch.Tensor:
        context_position = torch.arange(qlen, device=device)[:, None]
        memory_position = torch.arange(klen, device=device)[None, :]
        relative_position = memory_position - context_position
        rp_index = relative_position + (self.max_len - 1)
        rp_index = rp_index.clamp(0, 2 * self.max_len - 2)
        values = self.bias_table[rp_index.view(-1)].view(qlen, klen, self.num_heads)
        values = values.permute(2, 0, 1).contiguous()
        return values

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, max_rel_len: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.rel_pos = RelativePositionBias(max_len=max_rel_len, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        q = self.norm1(x)
        in_proj_weight = self.attn.in_proj_weight
        in_proj_bias = self.attn.in_proj_bias
        out_proj_weight = self.attn.out_proj.weight
        out_proj_bias = self.attn.out_proj.bias
        qkv = F.linear(q, in_proj_weight, in_proj_bias)
        q_, k_, v_ = qkv.chunk(3, dim=-1)
        head_dim = D // self.num_heads
        q_ = q_.view(B, S, self.num_heads, head_dim).transpose(1, 2)
        k_ = k_.view(B, S, self.num_heads, head_dim).transpose(1, 2)
        v_ = v_.view(B, S, self.num_heads, head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(head_dim)
        rel_bias = self.rel_pos(qlen=S, klen=S, device=attn_scores.device)
        attn_scores = attn_scores + rel_bias.unsqueeze(0)
        attn_prob = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_prob, v_)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        attn_out = F.linear(attn_out, out_proj_weight, out_proj_bias)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ViT1DEncoder(nn.Module):
    def __init__(self, signal_length: int, patch_size: int, in_channels: int,
                 embed_dim: int = 128, depth: int = 6, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, max_rel_len: int = 1024, use_cls_token: bool = True):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(signal_length=signal_length, patch_size=patch_size,
                                            in_channels=in_channels, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + (1 if use_cls_token else 0), embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                    dropout=dropout, max_rel_len=max_rel_len)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.patch_embed(x)
        S = x.size(1)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if x.size(1) != self.pos_embed.size(1):
            if self.use_cls_token:
                cls = self.pos_embed[:, :1, :]
                pos_patches = self.pos_embed[:, 1:, :].transpose(1, 2)
                new_N = x.size(1) - 1
                pos_patches = F.interpolate(pos_patches, size=new_N, mode='linear', align_corners=False)
                pos_patches = pos_patches.transpose(1, 2)
                pos_embed = torch.cat((cls, pos_patches), dim=1)
            else:
                pos_patches = self.pos_embed.transpose(1, 2)
                new_N = x.size(1)
                pos_patches = F.interpolate(pos_patches, size=new_N, mode='linear', align_corners=False)
                pos_embed = pos_patches.transpose(1, 2)
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed.to(x.device)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class MAE1D(nn.Module):
    def __init__(self, encoder: ViT1DEncoder, decoder_embed_dim: int = 128, decoder_depth: int = 4, mask_ratio: float = 0.75):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.embed_dim = encoder.embed_dim
        self.num_patches = encoder.num_patches
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + (1 if encoder.use_cls_token else 0), decoder_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=decoder_embed_dim, num_heads=max(1, encoder.blocks[0].num_heads//2),
                                    mlp_ratio=4.0, dropout=0.0, max_rel_len=encoder.blocks[0].rel_pos.max_len)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        patch_size = encoder.patch_embed.patch_size
        in_channels = encoder.patch_embed.proj.in_channels
        self.reconstruction_head = nn.Linear(decoder_embed_dim, patch_size * in_channels)
        
    def random_masking(self, x: torch.Tensor):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]
        # use gather to select kept tokens efficiently: prepare index for gather
        batch_indices = torch.arange(B, device=x.device)[:, None].expand(-1, len_keep)  # [B, len_keep]
        x_masked = x[batch_indices, ids_keep]  # [B, len_keep, D]
        # build mask (1 means masked) and then unshuffle it to original order
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)  # [B, N]
        return x_masked, mask, ids_restore, ids_keep
    
    def forward(self, signals: torch.Tensor):
        B = signals.size(0)
        patches = self.encoder.patch_embed(signals)  # [B, N, D]
        x_masked, mask, ids_restore, ids_keep = self.random_masking(patches)
        # encoder input
        if self.encoder.use_cls_token:
            cls_tokens = self.encoder.cls_token.expand(B, -1, -1).to(signals.device)
            encoder_input = torch.cat([cls_tokens, x_masked], dim=1)
            # positional interpolation
            pos = self.encoder.pos_embed
            if pos.size(1) != encoder_input.size(1):
                cls_p = pos[:, :1, :]
                pos_patches = pos[:, 1:, :].transpose(1,2)
                new_N = encoder_input.size(1) - 1
                pos_patches = F.interpolate(pos_patches, size=new_N, mode='linear', align_corners=False)
                pos_patches = pos_patches.transpose(1,2)
                pos_current = torch.cat((cls_p, pos_patches), dim=1).to(signals.device)
            else:
                pos_current = pos.to(signals.device)
            enc_x = encoder_input + pos_current
        else:
            encoder_input = x_masked
            pos = self.encoder.pos_embed
            if pos.size(1) != encoder_input.size(1):
                pos_patches = pos.transpose(1,2)
                new_N = encoder_input.size(1)
                pos_patches = F.interpolate(pos_patches, size=new_N, mode='linear', align_corners=False)
                pos_current = pos_patches.transpose(1,2).to(signals.device)
            else:
                pos_current = pos.to(signals.device)
            enc_x = encoder_input + pos_current
        x = self.encoder.dropout(enc_x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        if self.encoder.use_cls_token:
            encoded_patches = x[:, 1:, :]
        else:
            encoded_patches = x
        dec_in_vis = self.decoder_embed(encoded_patches)
        
        # create mask tokens
        Nvis = dec_in_vis.size(1)
        Nmask = self.num_patches - Nvis
        mask_tokens = self.mask_token.expand(B, Nmask, -1).to(signals.device)
        # concatenate visible and mask tokens (in kept+masked order) then unshuffle using ids_restore
        dec_tokens_cat = torch.cat([dec_in_vis, mask_tokens], dim=1)  # [B, num_patches, D_dec]
        # prepare index to scatter dec_tokens_cat into restored order
        ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, dec_tokens_cat.size(-1))  # [B, num_patches, D_dec]
        # create empty tensor and scatter
        dec_tokens = torch.zeros(B, self.num_patches, dec_tokens_cat.size(-1), device=signals.device, dtype=dec_tokens_cat.dtype)
        # For scatter, we need per-batch indexing via scatter_(1, index, src)
        dec_tokens = dec_tokens.scatter(1, ids_restore, dec_tokens_cat)
        # optionally prepend cls token for decoder
        if self.encoder.use_cls_token:
            cls_dec = self.decoder_pos_embed[:, :1, :].expand(B, -1, -1).to(signals.device)
            dec_tokens = torch.cat([cls_dec, dec_tokens], dim=1)
        # add decoder positional embedding (interpolate if needed)
        if dec_tokens.size(1) != self.decoder_pos_embed.size(1):
            if self.encoder.use_cls_token:
                cls = self.decoder_pos_embed[:, :1, :]
                pos_patches = self.decoder_pos_embed[:, 1:, :].transpose(1,2)
                new_N = dec_tokens.size(1) - 1
                pos_patches = F.interpolate(pos_patches, size=new_N, mode='linear', align_corners=False)
                pos_dec = torch.cat((cls, pos_patches.transpose(1,2)), dim=1).to(signals.device)
            else:
                pos_patches = self.decoder_pos_embed.transpose(1,2)
                new_N = dec_tokens.size(1)
                pos_dec = pos_patches
                pos_patches = F.interpolate(pos_patches, size=new_N, mode='linear', align_corners=False)
                pos_dec = pos_patches.transpose(1,2).to(signals.device)
        else:
            pos_dec = self.decoder_pos_embed.to(signals.device)
        dec_tokens = dec_tokens + pos_dec
        x_dec = dec_tokens
        for blk in self.decoder_blocks:
            x_dec = blk(x_dec)
        x_dec = self.decoder_norm(x_dec)
        if self.encoder.use_cls_token:
            x_rec = x_dec[:, 1:, :]
        else:
            x_rec = x_dec
        pred = self.reconstruction_head(x_rec)
        patch_size = self.encoder.patch_embed.patch_size
        in_channels = self.encoder.patch_embed.proj.in_channels
        pred = pred.view(B, self.num_patches, in_channels, patch_size)
        # original patches
        orig = signals
        orig_patches = orig.unfold(dimension=2, size=patch_size, step=patch_size).permute(0,2,1,3).contiguous()
        # compute MSE per patch and mask-weight it
        loss_per_patch = (pred - orig_patches).pow(2).mean(dim=(2,3))  # [B, num_patches]
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1.0)
        return {'loss': loss, 'pred': pred, 'mask': mask, 'orig_patches': orig_patches}

class ViT1DForClassification(nn.Module):
    def __init__(self, encoder: ViT1DEncoder, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, max(8, self.encoder.embed_dim//2)),
            nn.ReLU(),
            nn.Linear(max(8, self.encoder.embed_dim//2), num_classes)
        )
    def forward(self, signals: torch.Tensor):
        x = self.encoder(signals)
        if self.encoder.use_cls_token:
            patch_tokens = x[:, 1:, :]
        else:
            patch_tokens = x
        pooled = patch_tokens.mean(dim=1)
        return self.head(pooled)

# Quick sanity check
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 2
    C = 1
    long_L = 2048
    patch_size = 16
    encoder = ViT1DEncoder(signal_length=long_L, patch_size=patch_size, in_channels=C,
                           embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0,
                           dropout=0.1, max_rel_len=1024, use_cls_token=True)
    mae = MAE1D(encoder=encoder, decoder_embed_dim=64, decoder_depth=2, mask_ratio=0.75)
    dummy_long = torch.randn(B, C, long_L)
    out = mae(dummy_long)
    print("MAE pretrain forward -> loss:", out['loss'].item(), "pred shape:", out['pred'].shape, "mask shape:", out['mask'].shape)
    short_L = 256
    encoder_ft = ViT1DEncoder(signal_length=short_L, patch_size=patch_size, in_channels=C,
                              embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0,
                              dropout=0.1, max_rel_len=1024, use_cls_token=True)
    # transfer matching params
    sd_pre = encoder.state_dict()
    sd_ft = encoder_ft.state_dict()
    for k,v in sd_pre.items():
        if k in sd_ft and sd_ft[k].shape == v.shape:
            sd_ft[k] = v.clone()
    encoder_ft.load_state_dict(sd_ft)
    model_ft = ViT1DForClassification(encoder=encoder_ft, num_classes=5)
    dummy_short = torch.randn(B, C, short_L)
    logits = model_ft(dummy_short)
    print("Fine-tune forward -> logits shape:", logits.shape)
