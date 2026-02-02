# models/epsnet/spectrum_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralCIB(nn.Module):

    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, num_concepts=8):
        super().__init__()
        self.num_concepts = num_concepts

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=embed_dim * 4, activation='gelu', dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.concept_tokens = nn.Parameter(torch.randn(1, num_concepts, embed_dim))

        self.compress_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_log_var = nn.Linear(embed_dim, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to allow backpropagation through sampling."""
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, feature_sequence: torch.Tensor, condition_vector=None):
        batch_size = feature_sequence.size(0)
        processed_sequence = self.transformer_encoder(feature_sequence)
        compressed_output, _ = self.compress_attention(
            query=self.concept_tokens.expand(batch_size, -1, -1),
            key=processed_sequence,
            value=processed_sequence
        )
        compressed_output = self.norm(compressed_output)
        mu = self.fc_mu(compressed_output)
        log_var = self.fc_log_var(compressed_output)
        z = self.reparameterize(mu, log_var)

        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / batch_size 
        
        return z, kl_loss


class CNNBranch(nn.Module):
    """
    CNN branch for processing single spectrum (IR or Raman), implementing high-fidelity local feature extraction.
    """
    def __init__(self, cnn_out_channels=64, kernel_size=11, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, cnn_out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)

    def forward(self, x):
        features = F.relu(self.bn1(self.conv1(x)))
        features = F.relu(self.bn2(self.conv2(features)))
        return features


class Spectroformer(nn.Module):
    """
    New spectrum encoder implementing hierarchical, high-resolution feature extraction and abstraction.
    Flow: (CNN -> Concat) -> Patchify -> Project -> Transformer -> CIB
    """
    def __init__(self, config):
        super().__init__()
        ir_len = config.model.get('target_ir_len', 3500)
        raman_len = config.model.get('target_raman_len', 3500)
        
        cnn_out_channels = config.model.get('spec_cnn_out_channels', 64)
        cnn_kernel_size = config.model.get('spec_cnn_kernel_size', 11)
        cnn_stride = 2 
        num_cnn_layers_with_stride = 2
        concat_len = (ir_len // (cnn_stride**num_cnn_layers_with_stride)) + \
                     (raman_len // (cnn_stride**num_cnn_layers_with_stride))
        patch_size = config.model.get('spec_patch_size', 25)
        if concat_len % patch_size != 0:
            raise ValueError(f"Concatenated CNN feature length ({concat_len}) cannot be divisible by patch_size ({patch_size}).")
        num_patches = concat_len // patch_size
        patch_dim = cnn_out_channels * patch_size
        embed_dim = config.model.get('spec_embed_dim', 128)
        self.cnn_ir = CNNBranch(cnn_out_channels, cnn_kernel_size, cnn_stride)
        self.cnn_raman = CNNBranch(cnn_out_channels, cnn_kernel_size, cnn_stride)
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.patch_projection = nn.Linear(patch_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.cib = SpectralCIB(
            embed_dim=embed_dim,
            num_heads=config.model.spec_num_heads,
            num_layers=config.model.spec_num_layers,
            num_concepts=config.model.spec_num_concepts
        )

    def forward(self, batch, condition_vector=None):
        ir_spec = batch.ir_spectrum.view(batch.num_graphs, 1, -1)
        raman_spec = batch.raman_spectrum.view(batch.num_graphs, 1, -1)
        ir_features = self.cnn_ir(ir_spec)       
        raman_features = self.cnn_raman(raman_spec) 
        concat_features = torch.cat([ir_features, raman_features], dim=2)
        concat_features = concat_features.permute(0, 2, 1)
        B, L, C = concat_features.shape
        patches = concat_features.view(B, self.num_patches, self.patch_size, C)
        patches_flattened = patches.flatten(2)
        patch_embeddings = self.patch_projection(patches_flattened)
        patch_embeddings = patch_embeddings + self.position_embedding
        spectral_concepts, kl_loss = self.cib(patch_embeddings, condition_vector)
        
        return spectral_concepts, kl_loss