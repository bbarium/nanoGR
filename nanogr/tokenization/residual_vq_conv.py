from typing import Tuple

import torch
from torch.types import Tensor
from torch import nn
from torch.nn import functional as F

from nanogr.utils import kmeans


class MLP(nn.Module):
    def __init__(self, layers: list, dropout: float=0.0):
        super(MLP, self).__init__()
        
        convs = []
        for i in range(0, len(layers)-1):
            convs.append(nn.Dropout(p=dropout))
            convs.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                convs.append(nn.ReLU())

        self.mlp = nn.Sequential(*convs)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class VectorQuantization(nn.Module):
    def __init__(self, codebook_size: int, hid_channels: int, beta: float):
        super().__init__()
        self.codebook_size = codebook_size
        self.hid_channels = hid_channels
        self.beta = beta
        self.is_initialized = False
        self.utilization = None

        self.centroids = nn.Embedding(codebook_size, hid_channels)
        self.centroids.weight.data.zero_()
        
    def init_emb(self, data):
        centers = kmeans(data, self.codebook_size)
        self.centroids.weight.data.copy_(centers)
        self.is_initialized = True

    def forward(self, x: Tensor):
        latent = x.view(-1, self.hid_channels)
        
        if not self.is_initialized and self.training:
            self.init_emb(latent)
        
        distances = torch.cdist(latent, self.centroids.weight)
        indices = torch.argmin(distances, dim=-1)
        quantized = self.centroids(indices).view(x.shape)

        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = codebook_loss + self.beta * commitment_loss
        quantized = x + (quantized - x).detach()

        if self.training:
            self.utilization = round(len(torch.unique(indices.detach()))/self.codebook_size, 4) # Update codebook utilization
        return quantized, loss, indices


class ResidualVQ(nn.Module):
    def __init__(self, codebook_size: int, num_codebooks: int, hid_channels: int, beta: float):
        super().__init__()
        self.codebook_size = codebook_size
        
        self.vq_convs = nn.ModuleList()
        for _ in range(num_codebooks):
            self.vq_convs.append(VectorQuantization(codebook_size, hid_channels, beta))

    @property
    def utilization(self):
        return [vq.utilization for vq in self.vq_convs]

    def forward(self, x: Tensor):
        token_indices = []
        quantization_loss = 0
        residual, quantized_out = x, 0
        for vq_conv in self.vq_convs:
            quantized, loss, indices = vq_conv(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            
            token_indices.append(indices)
            quantization_loss += loss

        quantization_loss = quantization_loss / len(self.vq_convs)
        indices = torch.stack(token_indices, dim=-1)
        return quantized_out, quantization_loss, indices


class RQVAE(nn.Module):
    def __init__(self,
        in_channels: int,
        hid_channels: int,
        num_codebooks: int,
        codebook_size: int,
        mlp_channels: list=[256, 128, 64],
        dropout: float=0.0,
        quant_loss_weight: float=1.0,
        enable_kmeans_init: bool=True,
        kmeans_iters: int=100,
        beta: float=0.25,
        enable_norm: bool=False
    ):
        super(RQVAE, self).__init__()
        self.codebook_size = codebook_size
        self.quant_loss_weight = quant_loss_weight
        self.enable_norm = enable_norm
        self.enable_kmeans_init = enable_kmeans_init
        self.kmeans_iters = kmeans_iters
        
        channels = [in_channels] + mlp_channels + [hid_channels]
        self.encoder = MLP(layers=channels, dropout=dropout)
        self.decoder = MLP(layers=channels[::-1], dropout=dropout)

        self.rq = ResidualVQ(codebook_size, num_codebooks, hid_channels, beta)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    @torch.no_grad()
    def init_codebook(self, init_feat: Tensor):
        z = self.encoder(init_feat)
        
        residual = z
        for i in range(len(self.rq.vq_convs)):
            if not self.rq.vq_convs[i].is_initialized:
                if self.enable_kmeans_init:
                    centers = kmeans(residual, self.codebook_size, self.kmeans_iters)
                    self.rq.vq_convs[i].centroids.weight.data.copy_(centers)

                    quantized, _, _ = self.rq.vq_convs[i](residual)
                    residual = residual - quantized
                else:
                    raise NotImplementedError
                self.rq.vq_convs[i].is_initialized = True

    @torch.no_grad()
    def get_indices(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        _, _, indices = self.rq(x)
        return indices

    def get_utilization(self):
        return self.rq.utilization

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.encoder(x)
        embd, rq_loss, indices = self.rq(x)
        out = self.decoder(embd)
        return out, rq_loss, indices

    def compute_loss(self, x_recon: Tensor, x: Tensor, loss_quant: Tensor):
        loss_recon = F.mse_loss(x_recon, x, reduction='mean')
        loss_total = loss_recon + self.quant_loss_weight * loss_quant
        return loss_total, loss_recon, loss_quant