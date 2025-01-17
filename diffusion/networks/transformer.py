# code adapted from https://github.com/apapiu/transformer_latent_diffusion/

import torch
from einops.layers.torch import Rearrange
from torch import nn
from einops import rearrange
import gin


class PositionalEmbedding(nn.Module):

    def __init__(
        self,
        num_channels: int,
        max_positions: int,
        factor: float,
        endpoint: bool = False,
        rearrange: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.factor = factor
        self.rearrange = (Rearrange("b (f c) -> b (c f)", f=2)
                          if rearrange else nn.Identity())

    def forward(self, x: torch.Tensor):
        x = x.view(-1)
        x = x * self.factor
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            device=x.device,
        ).float()
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions)**freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return self.rearrange(x)


class MHAttention(nn.Module):

    def __init__(self, is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):
        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [
            rearrange(x, "bs n (h d) -> bs h n d", h=self.n_heads)
            for x in [q, k, v]
        ]

        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
            dropout_p=self.dropout_level if self.training else 0,
        )

        out = rearrange(out, "bs h n d -> bs n (h d)", h=self.n_heads)

        return out


class SelfAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 is_causal=False,
                 dropout_level=0.0,
                 n_heads=4):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v)


class CrossAttention(nn.Module):

    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.kv_linear = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x, y):
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q, k, v)


class MLP(nn.Module):

    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSepConv(nn.Module):

    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        """see: https://github.com/ofsoundof/LocalViT"""
        super().__init__()
        self.mlp = nn.Sequential(
            # this Conv with kernel size 1 is equivalent to the Linear layer in a "regular" transformer MLP
            nn.Conv1d(embed_dim,
                      mlp_multiplier * embed_dim,
                      kernel_size=1,
                      padding="same"),
            nn.Conv1d(
                mlp_multiplier * embed_dim,
                mlp_multiplier * embed_dim,
                kernel_size=3,
                padding="same",
                groups=mlp_multiplier * embed_dim,
            ),  # <- depthwise conv
            nn.GELU(),
            nn.Conv1d(mlp_multiplier * embed_dim,
                      embed_dim,
                      kernel_size=1,
                      padding="same"),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        x = rearrange(x, "b t c -> b c t")
        x = self.mlp(x)
        x = rearrange(x, "b c t -> b t c")
        return x


class DecoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        cond_dim: int,
        use_crossattn: int,
        is_causal: bool,
        mlp_multiplier: int,
        dropout_level: float,
        mlp_class: type[MLP] | type[MLPSepConv],
    ):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim,
                                            is_causal,
                                            dropout_level,
                                            n_heads=embed_dim // 64)
        self.use_ca = use_crossattn
        if use_crossattn:
            self.cross_attention = CrossAttention(embed_dim,
                                                  is_causal=False,
                                                  dropout_level=0,
                                                  n_heads=embed_dim // 64)
            self.norm4 = nn.LayerNorm(embed_dim)
        self.mlp = mlp_class(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(cond_dim, 2 * embed_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                tcond: torch.Tensor | None) -> torch.Tensor:
        x = self.self_attention(self.norm1(x)) + x
        # AdaLN
        alpha, beta = self.linear(cond).chunk(2, dim=-1)
        x = self.norm2(x) * (1 + alpha.unsqueeze(1)) + beta.unsqueeze(1)
        # Cross-Attention if time conditioning is activated
        if self.use_ca:
            assert tcond is not None
            x = self.cross_attention(self.norm4(x), tcond) + x
        x = self.mlp(self.norm3(x)) + x
        return x


class DenoiserTransBlock(nn.Module):

    def __init__(
        self,
        n_channels: int = 64,
        seq_len: int = 32,
        mlp_multiplier: int = 4,
        embed_dim: int = 256,
        cond_dim: int = 128,
        tcond_dim: int = 0,
        dropout: float = 0.1,
        n_layers: int = 4,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        self.patchify_and_embed = nn.Sequential(
            Rearrange("b c t -> b t c"),
            nn.Linear(n_channels, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        use_crossattn = tcond_dim > 0
        if use_crossattn:
            self.patchify_and_embed_tcond = nn.Sequential(
                Rearrange("b c t -> b t c"),
                nn.Linear(tcond_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )
        else:
            self.patchify_and_embed_tcond = nn.Identity()

        self.rearrange2 = Rearrange("b t c -> b c t", )

        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer("precomputed_pos_enc",
                             torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim=self.embed_dim,
                mlp_multiplier=self.mlp_multiplier,
                cond_dim=cond_dim,
                use_crossattn=use_crossattn,
                is_causal=False,
                dropout_level=self.dropout,
                mlp_class=MLPSepConv,
            ) for _ in range(self.n_layers)
        ])

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, n_channels),
                                      self.rearrange2)

    def forward(self, x, cond, tcond):
        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
        x = x + self.pos_embed(pos_enc)
        for block in self.decoder_blocks:
            x = block(x, cond, self.patchify_and_embed_tcond(tcond))
        return self.out_proj(x)


@gin.configurable
class Denoiser(nn.Module):

    def __init__(
        self,
        n_channels: int = 64,
        seq_len: int = 32,
        embed_dim: int = 256,
        cond_dim: int = 64,
        tcond_dim: int = 0,
        noise_embed_dims: int = 64,
        n_layers: int = 6,
        mlp_multiplier: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels

        self.fourier_feats = PositionalEmbedding(num_channels=noise_embed_dims,
                                                 max_positions=10_000,
                                                 factor=100.0)

        self.embedding = nn.Sequential(
            nn.Linear(noise_embed_dims + cond_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.denoiser_trans_block = DenoiserTransBlock(
            n_channels=n_channels,
            seq_len=seq_len,
            mlp_multiplier=mlp_multiplier,
            embed_dim=embed_dim,
            dropout=dropout,
            n_layers=n_layers,
            cond_dim=self.embed_dim,
            tcond_dim=tcond_dim,
        )

    def forward(self, x, time, time_cond=None, cond=None):
        noise_level = self.fourier_feats(time)
        features = torch.cat([noise_level, cond], dim=-1)
        features = self.embedding(features)

        x = self.denoiser_trans_block(x, features, time_cond)
        return x


if __name__ == "__main__":

    x = torch.randn(3, 64, 32)
    tcond = torch.randn(3, 128, 32)
    n = torch.rand(3)
    model = Denoiser(tcond_dim=128)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Number of parameters: {n_params:.2f}M")
    out = model(x, n, tcond)
    print(out.shape)
