import os
import time
import json

import torch
from torch.types import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_constant_schedule_with_warmup

from nanogr.tokenization import AbstractTokenizer
from nanogr.utils import log

from .residual_vq import RQVAE


class RQVAETokenizer(AbstractTokenizer):
    def __init__(
        self,
        id2item: dict,
        num_codebooks: int,
        codebook_size: int,
        in_channels: int,
        hid_channels: int,
        mlp_channels: list,
        dropout: float,
        quant_loss_weight: float,
        beta: float,
        enable_kmeans_init: bool,
        kmeans_iters: int,
        lr: float,
        wd: float,
        epochs: int,
        num_warmup_steps: int,
        eval_step: int,
        batch_size: int,
        category: str,
        val_split: float=0.1,
        output_path: str='./outputs',
    ):
        super(RQVAETokenizer, self).__init__(num_codebooks, codebook_size, output_path)
        
        self.model = RQVAE(
            in_channels,
            hid_channels,
            num_codebooks,
            codebook_size,
            mlp_channels=mlp_channels,
            dropout=dropout,
            quant_loss_weight=quant_loss_weight,
            enable_kmeans_init=enable_kmeans_init,
            kmeans_iters=kmeans_iters,
            beta=beta,
        )
        log(self.model)
        
        # Hyperparameters of trainer
        self.lr = lr
        self.wd = wd
        self.num_warmup_steps = num_warmup_steps
        self.val_split = val_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_step = eval_step
        
        self.category = category
        self.hid_channels = hid_channels
        self.id2item = id2item

    @property
    def index_factory(self) -> str:
        return f'{self.category}_RQVAE{self.num_codebooks}x{self.codebook_size}.dim{self.hid_channels}'
    
    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        
        total_loss, total_loss_recon, total_loss_quant = 0, 0, 0
        for batch_data in train_loader:
            self.optimizer.zero_grad()
            x_recon, loss_rq, _ = self.model(batch_data)
            loss, loss_recon, loss_quant = self.model.compute_loss(x_recon, batch_data, loss_rq)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_loss_recon += loss_recon.item()
            total_loss_quant += loss_quant.item()
        return total_loss, total_loss_recon, total_loss_quant
    
    @torch.no_grad()
    def eval_epoch(self, train_embd: Tensor, val_embd: Tensor):
        self.model.eval()
        
        indices = self.model.get_indices(train_embd).detach().cpu()
        indices_uniq = torch.unique(indices, dim=0)
        collision_rate = 1 - (len(indices_uniq) / len(indices)) 
        utilization = self.model.get_utilization()
        
        val_embd_recon, loss_rq, _ = self.model(val_embd)
        total_val_loss, _, _ = self.model.compute_loss(val_embd_recon, val_embd, loss_rq)
        return indices, collision_rate, utilization, total_val_loss
    
    def _generate_semantic_id(self, sent_embd: Tensor, device: str) -> dict:
        writer = SummaryWriter()
        
        sent_embd = sent_embd.to(device)
        self.model = self.model.to(device)
        
        # Notice: the dataset splits (training/valid) are just used to evaluation the quality of codebooks. 
        # We will generate SIDs of all items for downstream generative recommandation tasks
        # val_split = int(sent_embd.size(0) * self.val_split)
        # indices_shuffled = torch.randperm(sent_embd.size(0))
        # val_embd, train_embd = sent_embd[indices_shuffled[:val_split], :], sent_embd[indices_shuffled[val_split:], :]
        val_embd, train_embd = sent_embd, sent_embd
        train_loader = DataLoader(train_embd, self.batch_size, shuffle=False)
        log(f'Train size: {train_embd.size(0)}, Val size: {val_embd.size(0)}')
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, self.num_warmup_steps*len(train_loader))
        
        best_loss, best_collision, PATIENCE = float('inf'), float('inf'), 0
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss, loss_recon, loss_quant = self.train_epoch(train_loader)

            indices, collision_rate, utilization, val_loss = self.eval_epoch(train_embd, val_embd)
            if train_loss < best_loss or collision_rate < best_collision:
                best_loss = train_loss
                best_collision = collision_rate
                PATIENCE = 0
                
                item2sid = {}
                for i in range(indices.shape[0]):
                    item = self.id2item[i + 1]
                    item2sid[item] = tuple(indices[i].tolist())
            else:
                PATIENCE += 1
            end_time = time.time()
            writer.add_scalar('Train/Total Loss', train_loss, global_step=epoch)
            writer.add_scalar('Train/Reconstruction Loss', loss_recon, global_step=epoch)
            writer.add_scalar('Train/Quantization Loss', loss_quant, global_step=epoch)
            writer.add_scalar('Train/Val Loss', val_loss, global_step=epoch)
            writer.add_scalar('Train/Collision Rate', collision_rate, global_step=epoch)
            
            if epoch == 1 or epoch % self.eval_step == 0:
                log(
                    f'Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ' \
                    f'Collision Rate: {collision_rate:.2%}, Utilization: {utilization}, Elapse Time (s): {end_time-start_time:.2f}'
                )
            if PATIENCE >= 99999:
                break
            
        writer.close()
        return (
            item2sid, 
            [self.model.rq.vq_convs[i].centroids.weight.detach().cpu() for i in range(len(self.model.rq.vq_convs))],
            collision_rate,
            utilization
        )
    
    def emb_to_sid(self, sent_embd: Tensor, device: str) -> dict:
        sem_ids_path = os.path.join(self.output_path, f'sentence-t5-base_{self.index_factory}.sem_ids')

        if not os.path.exists(sem_ids_path):
            item2sid, codebooks, collision_rate, utilization = self._generate_semantic_id(sent_embd, device)
            log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
            # with open(sem_ids_path, 'w') as f:
            #     json.dump(item2sid, f)
            # torch.save(codebooks, './outputs/codebooks.pt')
        else:
            log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
            item2sid = json.load(open(sem_ids_path, 'r'))


        # item2sid = self.collision_mitigation(item2sid, self.codebook_size) # Add an additional codebook to mitigate the hash collision
        
        # item2sid = self.sid_shift(item2sid)
        return item2sid, collision_rate, utilization
    