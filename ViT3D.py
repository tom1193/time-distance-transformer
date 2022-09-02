
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils import *
from os.path import join as pjoin
from ViT import Transformer, LearnablePatchPosition, AbsTimeEncoding

def triple(t):
    return t if isinstance(t, tuple) else (t, t, t)

class ConvPatchEmbedding(nn.Module):
    def __init__(self, patch_height, patch_width, patch_depth, channels, dim):
        super().__init__()
        self.embedding = nn.Sequential(
            Rearrange('b t n (p1 p2 p3 c) -> (b t n) c p1 p2 p3', p1=patch_height, p2=patch_width, p3=patch_depth),
            nn.Conv3d(channels, 128, 3, stride=2, padding=1), # divies patch size by two
            nn.GELU(),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(256, dim, kernel_size=(patch_height//4, patch_width//4, patch_depth//4), 
                stride=(patch_height//4, patch_width//4, patch_depth//4)),
        )
    def forward(self, x):
        b, t, *_ = x.shape
        x = self.embedding(x)
        x = rearrange(x, '(b t n) c p1 p2 p3 -> b t n (p1 p2 p3 c)', b=b, t=t)
        return x

class SimpleViT3D(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, qkv_bias=False,
                 time_embedding="PositionalEncoding", pos_embedding="LearnablePatchPosition", 
                 patch_embedding="Linear", channels=1, dim_head=64, dropout=0.1, phase="train"):
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
        image_height, image_width, image_depth = triple(image_size)
        patch_height, patch_width, patch_depth = triple(patch_size)

        assert image_height % patch_height == 0 \
            and image_width % patch_width == 0 \
            and image_depth % patch_depth == 0, 'Image dimensions must be divisible by the patch size.'
        patched_height, patched_width, patched_depth = ((image_height // patch_height), 
                                                        (image_width // patch_width),
                                                        (image_depth // patch_depth))
        num_patches = patched_height * patched_width * patched_depth
        print(f"Number of patches: {num_patches}")
        self.patch_dim = channels * patch_height * patch_width * patch_depth

        if patch_embedding=="Linear":
            print("Linear patch embedding")
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b t c (h p1) (w p2) (z p3) -> b t (h w z) (p1 p2 p3 c)', p1=patch_height, p2=patch_width, p3=patch_depth),
                nn.Linear(self.patch_dim, dim),
            )
        elif patch_embedding=="Conv":
            print("Conv patch embedding")
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b t c (h p1) (w p2) (z p3) -> b t (h w z) (p1 p2 p3 c)', p1=patch_height, p2=patch_width, p3=patch_depth),
                ConvPatchEmbedding(patch_height, patch_width, patch_depth, channels, dim),
            )
        
        # different types of positional embeddings
        # 1. PositionalEncoding: Fixed alternating sin cos with position
        # 2. AbsTimeEmb: Fixed alternating sin cos with time
        # 3. Learnable: self.time_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        time_emb_dict = {
            # "PositionalEncoding": PositionalTimeEncoding,
            "AbsTimeEncoding": AbsTimeEncoding,
            # "LearnableEmb": LearnableEmb,
        }
        self.time_embedding = time_emb_dict[time_embedding](dim, 0.1, num_patches=num_patches)

        pos_emb_dict = {
            # "PatchPosition2D": PatchPosition2D,
            "LearnablePatchPosition": LearnablePatchPosition,
        }
        self.pos_embedding = pos_emb_dict[pos_embedding](num_patches, dim)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, qkv_bias, dropout)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            # nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img, times):
        # b, t, _, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img) # (b t p d)
        x = self.pos_embedding(x) # (b t p d)
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
