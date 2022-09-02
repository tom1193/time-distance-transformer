"""
Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py that implements 
Kaiming He et al. Masked Autoencoders Are Scalable Vision Learners (https://arxiv.org/pdf/2111.06377.pdf)
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ViT import Transformer, AbsTimeEncoding
from FactorizedViT3D import FactorizedViT3D

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        encoder_class,
        decoder_dim,
        masking_ratio = 0.6,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        self.encoder_class = encoder_class
        num_patches, encoder_dim = encoder.pos_embedding.pe.shape
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = encoder.patch_dim
        self.num_masked = int(masking_ratio * num_patches)
        self.encoder_time_emb = AbsTimeEncoding(encoder_dim, dropout=0.1, num_patches=num_patches-self.num_masked)

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, 
                                   dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4, qkv_bias=False)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.decoder_time_emb = AbsTimeEncoding(decoder_dim, dropout=0.1, num_patches=num_patches-self.num_masked)
        self.masked_decoder_time_emb = AbsTimeEncoding(decoder_dim, dropout=0.1, num_patches=self.num_masked)
        
        if encoder_dim == decoder_dim:
            with torch.no_grad():
                # copy encoder's positional embeddings to decoder's
                self.decoder_pos_emb.weight.copy_(self.encoder.pos_embedding.pe)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img, times):
        device = img.device

        # get patches
        patches = self.to_patch(img)
        # batch, num_patches, *_ = patches.shape
        batch, t, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        # tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        tokens = self.encoder.pos_embedding(tokens) # (b t p d)

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        rand_indices = torch.rand(batch, t, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :, :self.num_masked], rand_indices[:, :, self.num_masked:]
        
        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None, None]
        time_range = torch.arange(t, device=device)[:, None]
        # unmasked_indexer = repeat(unmasked_indices, 'b m -> b t m', t=t)
        # tokens = tokens[batch_range, unmasked_indices] # expects (b p d)
        tokens = tokens[batch_range, time_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        patches = rearrange(patches, 'b t ... d -> b t (...) d')
        # masked_indexer = repeat(masked_indices, 'b m -> b t m', t=t)
        masked_patches = patches[batch_range, time_range, masked_indices]

        # attend with vision transformer
        if self.encoder_class=="FactorizedViT3D":
            # only interactions between tokens of the same temporal index are modeled
            *_, p, d = tokens.shape
            encoded_tokens = torch.zeros(batch, t, p, d, dtype=torch.float32, device=device)
            for i in range(t):
                encoded_tokens[:, i] = self.encoder.encoder1(tokens[:, i])
            encoded_tokens = rearrange(encoded_tokens, 'b t ... d -> b (t ...) d')
        elif self.encoder_class=="TimeDistanceViT":
            # distance matrix
            R = torch.zeros(batch, t, t, device=device, dtype=torch.float32)
            for n in range(batch):
                for i in range(t):
                    for j in range(t):
                        R[n, i, j] = torch.abs(times[n, i] - times[n, j])

            tokens = rearrange(tokens, 'b t ... d -> b (t ...) d')
            tokens = self.encoder_time_emb(tokens, times)
            encoded_tokens = self.encoder.transformer(tokens, R)
        else:
            # interactions between all tokens are modeled
            tokens = rearrange(tokens, 'b t ... d -> b (t ...) d')
            tokens = self.encoder_time_emb(tokens, times)
            encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply position and time embeddings to unmasked tokens
        unmasked_pe = rearrange(self.decoder_pos_emb(unmasked_indices), 'b t m d -> b (t m) d', t=t)
        decoder_tokens = decoder_tokens + unmasked_pe
        decoder_tokens = self.decoder_time_emb(decoder_tokens, times)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b (t n) d', b = batch, n = self.num_masked, t=t)
        masked_pe = rearrange(self.decoder_pos_emb(masked_indices), 'b t m d -> b (t m) d', t=t)
        mask_tokens = mask_tokens + masked_pe
        mask_tokens = self.masked_decoder_time_emb(mask_tokens, times)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :(self.num_masked*t)]
        pred_pixel_values = self.to_pixels(mask_tokens)
        pred_pixel_values = repeat(pred_pixel_values, "b (t p) d -> b t p d", t=t)
        
        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss

# debugging
# if __name__ == "__main__":
#     vit = FactorizedViT3D(
#         image_size=256,
#         patch_size=32,
#         num_classes=100,
#         dim=2048,
#         depth=4,
#         heads=8,
#         mlp_dim=4096,
#         qkv_bias=False,
#         time_embedding="AbsTimeEncoding",
#         pos_embedding="LearnablePatchPosition",
#         dim_head=64,
#     )
#     mae = MAE(
#         encoder=vit,
#         encoder_class="FactorizedViT3D",
#         masking_ratio=0.75,
#         decoder_dim=256,
#         decoder_depth=4
#     )
#     data = torch.rand(3, 2, 1, 256, 256, 256, dtype=torch.float32)
#     times = torch.rand(3, 2, dtype=torch.float32)
#     loss = mae(data, times)
