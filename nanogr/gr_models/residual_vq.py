import torch
from torch.types import Tensor
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, layers: list, dropout: float=0.0):
        super(MLP, self).__init__()
        
        convs = [nn.Linear(layers[0], layers[1])] # in_channels -> hid_channels
        for i in range(1, len(layers)-1):
            convs.append(nn.ReLU())
            convs.append(nn.Linear(layers[i], layers[i+1]))
            convs.append(nn.Dropout(p=dropout))

        self.model = nn.Sequential(*convs)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature) -> Tensor:
        return self.model(input_feature)


class VectorQuantization(nn.Module):
    def __init__(
        self, 
        codebook_size: int,
        hid_channels: int,
        enable_kmeans_init: bool,
        kmeans_iters: int,
        beta: float,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.hid_channels = hid_channels
        self.beta = beta
        self.kmeans_iters = kmeans_iters
        self.enable_kmeans_init = enable_kmeans_init
        self.is_initialized = False

        self.centroids = nn.Parameter(torch.zeros(codebook_size, hid_channels), requires_grad=True)

    def forward(self, x):
        embd_tmp = x
        # if not self.is_initialized and self.training:
        #     self.initialization(embd_tmp)

        # TODO：这个Eulidean距离度量还需要再修改
        d = torch.sum(embd_tmp**2, dim=1, keepdim=True) + \
            torch.sum(self.centroids**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(embd_tmp, self.centroids.t())
        indices = torch.argmin(d, dim=-1)
        embd = self.centroids[indices]

        commitment_loss = F.mse_loss(embd.detach(), x)
        codebook_loss = F.mse_loss(embd, x.detach())
        loss = codebook_loss + self.beta * commitment_loss
        
        embd = x + (embd - x).detach()
        return embd, loss, indices


class ResidualVQ(nn.Module):
    def __init__(self, codebook_sz_list: list, hid_channels: int, enable_kmeans_init: int, kmeans_iters: int, beta: float):
        super().__init__()
        self.vq_convs = nn.ModuleList()
        for codebook_size in codebook_sz_list:
            self.vq_convs.append(VectorQuantization(codebook_size, hid_channels, enable_kmeans_init, kmeans_iters, beta))

    def forward(self, x: Tensor):
        token_indices = []
        quantization_loss = torch.tensor(0.0).to(x.device)
        embd_residual, embd_quantized = x, torch.zeros_like(x)
        for idx in range(len(self.vq_convs)):
            # flag = True
            # if self.vq_convs[idx].is_initialized and not self.vq_convs[-1].is_initialized:
            #     flag = False
            # elif idx == 0:
            #     flag = True
            # elif self.vq_convs[idx].is_initialized or self.vq_convs[idx].is_initial_step:
            #     flag = True
            
            # if flag:
            # else:
            #     token_indices, embd = self.vq_convs[idx].predict_indices(embd_residual)
            
            embd, loss, indices = self.vq_convs[idx](embd_residual)
            quantization_loss += loss
            
            embd_residual = embd_residual - embd
            embd_quantized = embd_quantized + embd
            token_indices.append(indices)

        token_indices = torch.stack(token_indices, dim=-1)
        return embd_quantized, quantization_loss, token_indices


class RQVAE(nn.Module):
    def __init__(self,
        hid_channels: int,
        num_codebooks: int,
        codebook_size: int,
        mlp_channels: list=[768, 256, 128, 64],
        dropout: float=0.0,
        loss_type: str="mse",
        quant_loss_weight: float=1.0,
        enable_kmeans_init: bool=True,
        kmeans_iters: int=100,
        beta: float=0.25,
    ):
        super(RQVAE, self).__init__()
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight

        self.encoder = MLP(layers=mlp_channels, dropout=dropout)
        self.decoder = MLP(layers=mlp_channels[::-1], dropout=dropout)
        self.bn = nn.BatchNorm1d(hid_channels)

        codebook_sz_list = [codebook_size] * num_codebooks
        self.rq = ResidualVQ(codebook_sz_list, hid_channels, enable_kmeans_init, kmeans_iters, beta)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.bn(x)
        embd, rq_loss, indices = self.rq(x)
        out = self.decoder(embd)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, x):
        x = self.encoder(x)
        x = self.bn(x)
        _, _, indices = self.rq(x)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total
