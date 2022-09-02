from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils import *
from os.path import join as pjoin
from ViT import Transformer, AbsTimeEncoding, PositionalTimeEncoding
from TimeDistanceViT import TimeAwareTransformer

class FeatViT(nn.Module):
    def __init__(self, *, num_feat, feat_dim, num_classes, dim, depth, heads, mlp_dim, qkv_bias=False,
                 time_embedding="PositionalEncoding", dim_head=64, dropout=0.1):
        """
        num_feat: number of input features
        feat_dim: dimension of input features
        dim: dimension after linear projection of input
        depth: number of Transformer blocks
        heads: number of heads in multi-headed attention
        mlp_dim: dimension of MLP layer in each Transformer block
        qkv_bias: enable bias for qkv if True
        seq_length: number of items in the sequence
        dim_head: dimension of q,k,v,z
        """
        super().__init__()
        self.feat_embedding = nn.Sequential(
            # CHANGE: added time dimension
            nn.Linear(feat_dim, dim),
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
        self.time_embedding = time_emb_dict[time_embedding](dim, 0.1, seq_len=2, num_patches=num_feat)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, feat, times):
        # b, t, n, d = img.shape

        x = self.feat_embedding(feat)
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


class TimeAwareFeatViT(nn.Module):
    def __init__(self, *, num_feat, feat_dim, num_classes, dim, depth, heads, mlp_dim, qkv_bias=False,
                 time_embedding="PositionalEncoding", pos_embedding=None, dim_head=64, dropout=0.1):
        """
        num_feat: number of input features
        feat_dim: dimension of input features
        dim: dimension after linear projection of input
        depth: number of Transformer blocks
        heads: number of heads in multi-headed attention
        mlp_dim: dimension of MLP layer in each Transformer block
        qkv_bias: enable bias for qkv if True
        seq_length: number of items in the sequence
        dim_head: dimension of q,k,v,z
        """
        super().__init__()
        self.feat_embedding = nn.Sequential(
            nn.Linear(feat_dim, dim),
        )

        self.transformer = TimeAwareTransformer(dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, feat, times):
        b, t, _, d = feat.shape

        x = self.feat_embedding(feat)
        x = rearrange(x, 'b t ... d -> b (t ...) d')

        # Create distance matrix from times
        # R = torch.zeros(b, t, t, device=x.device, dtype=torch.float32)
        # for n in range(b):
        #     for i in range(t):
        #         for j in range(t):
        #             R[n, i, j] = torch.abs(times[n, i] - times[n, j])

        # Create distance matrix from times
        # rows i maps to key token i, column j maps to query token j
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
    model = FeatViT(
        num_feat=5,
        feat_dim=64,
        num_classes=2,
        dim=64,
        depth=8,
        heads=8,
        mlp_dim=256,
        qkv_bias=False,
        time_embedding="PositionalEncoding",
    )
    data = torch.rand(3, 2, 5, 64)
    times = torch.rand(3, 2)
    output = model(data, times)
    a = 2