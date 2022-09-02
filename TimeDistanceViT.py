import math
import torch
from torch import nn
from torch.autograd import Variable

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils import *
from os.path import join as pjoin

from ViT import PatchPosition2D, LearnablePatchPosition
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def patch_posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, t, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class PositionalTimeEncoding(nn.Module):
    # Fixed time-invariant positional encoding
    def __init__(self, dim, dropout=0.1, seq_len=5, num_patches=1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(seq_len*num_patches, dim)
        position = torch.arange(0, seq_len).repeat_interleave(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, t):
        x = x + Variable(self.pe, requires_grad=False)
        return self.dropout(x)

class AbsTimeEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, num_patches=1):
        super().__init__()
        self.num_patches = num_patches
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(torch.arange(0, dim, 2) *-(math.log(10000.0) / dim))
        self.register_buffer('div_term', div_term)
        self.dim = dim
        
    def forward(self, x, t):
        device, dtype = x.device, x.dtype
        pe = torch.zeros(x.shape, device=device, dtype=dtype)
        
        # repeat times into shape [b, t, dim]
        time_position = repeat(t, 'b t -> b t d', d=int(self.dim/2))
        time_position = time_position.repeat_interleave(self.num_patches, dim=1)
        pe[:, :, 0::2] = torch.sin(time_position * self.div_term.expand_as(time_position))
        pe[:, :, 1::2] = torch.cos(time_position * self.div_term.expand_as(time_position))
        x = x + Variable(pe, requires_grad=False)
        return self.dropout(x)

class TimeAwareV1(nn.Module):
    """
    V1 models time distance as a flipped sigmoid function
    Learnable parameters for each attention head:
        self.a describes slope of decay
        self.c describes position of decay
    """

    def __init__(self, heads=8):
        super().__init__()
        self.heads = heads
        # initialize from [0,1], which fits decay to scale of fractional months
        self.a = nn.Parameter(torch.rand(heads), requires_grad=True)
        # initialize from [0,12], which fits position to the scale of fractional months
        self.c = nn.Parameter(12 * torch.rand(heads), requires_grad=True)

    def forward(self, x, R):
        *_, n = x.shape
        b, _, t = R.shape
        num_patches = int(n/t)
        # repeat R for each head
        R = repeat(R, 'b t1 t2 -> b h t1 t2', h=self.heads)
        # repeat parameters
        a = repeat(self.a, 'h -> b h t1 t2', b=b, t1=t, t2=t)
        c = repeat(self.c, 'h -> b h t1 t2', b=b, t1=t, t2=t)
        # flipped sigmoid with learnable parameters
        R = 1 / (1 + torch.exp(torch.abs(a) * R - torch.abs(c)))
        # repeat values along last two dimensions according to number of patches
        R = R.repeat_interleave(num_patches, dim=2)
        R = R.repeat_interleave(num_patches, dim=3)
        return x * R


class TimeAwareAttention(nn.Module):
    """
    Implements distance aware attention weight scaling from https://arxiv.org/pdf/2010.06925.pdf#cite.yan2019tener
    """
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.time_dist = TimeAwareV1(heads=heads)
        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, R):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        time_scaled_dots = self.time_dist(self.relu(dots), R)

        attn = self.attend(time_scaled_dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)


class TimeAwareTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TimeAwareAttention(dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x, R):
        for attn, ff in self.layers:
            x = attn(x, R) + x
            x = ff(x) + x
        return x


class TimeDistanceViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, qkv_bias=False,
                 time_embedding="PositionalEncoding", pos_embedding=None,
                 seq_length=5, channels=3, dim_head=64, dropout=0.1):
        """
        if image_size = patch_size then does not split into patch and max sequence length = seq_length
        dim: dimension after linear projection of input
        depth: number of Transformer blocks
        heads: number of heads in multi-headed attention
        mlp_dim: dimension of MLP layer in each Transformer block
        seq_length: number of items in the sequence
        dim_head: dimension of q,k,v,z
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        print(f"Number of patches: {num_patches}")
        self.patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            # CHANGE: added time dimension
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, dim),
        )

        self.time_embedding = PositionalTimeEncoding(dim, 0.1, num_patches=num_patches)
        
        pos_emb_dict = {
            "PatchPosition2D": PatchPosition2D,
            "LearnablePatchPosition": LearnablePatchPosition,
        }
        self.pos_embedding = pos_emb_dict[pos_embedding](num_patches, dim)

        self.transformer = TimeAwareTransformer(dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, times):
        b, t, _, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        x = self.pos_embedding(x)
        x = rearrange(x, 'b t ... d -> b (t ...) d')

        # Create distance matrix from times
        R = torch.zeros(b, t, t, device=x.device, dtype=torch.float32)
        for n in range(b):
            for i in range(t):
                for j in range(t):
                    R[n, i, j] = torch.abs(times[n, 0] - times[n, i])

        x = self.transformer(x, R)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)

    def load_from_mae(self, weights):
        # load all weights from pretrained MAE except linear head
        pretrained_headless = {k: v for k, v in weights.items() if HEAD not in k}
        model_state = self.state_dict()
        model_state.update(pretrained_headless)
        self.load_state_dict(model_state)


if __name__ == "__main__":
    model = TimeDistanceViT(
        image_size=32,
        patch_size=8,
        num_classes=2,
        dim=64,
        depth=8,
        heads=8,
        mlp_dim=256,
        qkv_bias=False,
        time_embedding=None,
        pos_embedding="PatchPosition2D",
    )
    data = torch.rand(3, 5, 3, 32, 32)
    times = torch.rand(3, 5)
    output = model(data, times)
    print(output.shape)
    a = 2
