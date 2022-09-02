import math
from scipy import ndimage
import torch
from torch import nn
from torch.autograd import Variable

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils import *
from os.path import join as pjoin

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

class PatchPosition2D(nn.Module):
    def __init__(self, num_patches, dim, temperature=10000, dtype=torch.float32):
        super().__init__()
        # h is num of patches along height and width
        h = math.sqrt(num_patches)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(h), indexing='ij')
        assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1. / (temperature ** omega)
        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        self.pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)   
        
    def forward(self, x):
        b, t, *_ = x.shape
        p, d = self.pe.shape
        pe = self.pe.expand(b, t, p, d)
        return x + pe.to(x.device)

class LearnablePatchPosition(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(num_patches, dim), requires_grad=False)
    def forward(self, x):
        b, t, *_ = x.shape
        p, d = self.pe.shape
        pe = self.pe.expand(b, t, p, d)
        return x + pe.to(x.device)

class PositionalTimeEncoding(nn.Module):
    # Fixed time-invariant positional encoding
    def __init__(self, dim, dropout=0.1, seq_len=2, num_patches=1):
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
    def __init__(self, dim, dropout=0.1, seq_len=2, num_patches=1):
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

# class LearnableEmb(nn.Module):
#     def __init__(self, dim, dropout=0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         self.w0 = nn.Parameter(torch.rand())
#
#     def forward(self, x, t):

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

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, qkv_bias=qkv_bias, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, qkv_bias=False,
                 time_embedding="PositionalEncoding", pos_embedding="PatchPosition2D", 
                 seq_length=5, channels=3, dim_head=64, dropout=0.1):
        """
        if image_size = patch_size then does not split into patch and max sequence length = seq_length
        dim: dimension after linear projection of input
        depth: number of Transformer blocks
        heads: number of heads in multi-headed attention
        mlp_dim: dimension of MLP layer in each Transformer block
        qkv_bias: enable bias for qkv if True
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
        
        # different types of positional embeddings
        # 1. PositionalEncoding: Fixed alternating sin cos with position
        # 2. AbsTimeEmb: Fixed alternating sin cos with time
        # 3. Learnable: self.time_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        time_emb_dict = {
            "PositionalEncoding": PositionalTimeEncoding,
            "AbsTimeEncoding": AbsTimeEncoding,
            # "LearnableEmb": LearnableEmb,
        }
        self.time_embedding = time_emb_dict[time_embedding](dim, 0.1, num_patches=num_patches)

        pos_emb_dict = {
            "PatchPosition2D": PatchPosition2D,
            "LearnablePatchPosition": LearnablePatchPosition,
        }
        self.pos_embedding = pos_emb_dict[pos_embedding](num_patches, dim)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img, times):
        b, t, _, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        x = self.pos_embedding(x)
        x = rearrange(x, 'b t ... d -> b (t ...) d')

        x = self.time_embedding(x, times)
        
        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)
    
    def load_from_mae(self, weights):
        # load all weights from pretrained MAE except linear head
        pretrained_headless = {k: v for k, v in weights.items() if HEAD not in k}
        model_state = self.state_dict()
        model_state.update(pretrained_headless)
        self.load_state_dict(model_state)

