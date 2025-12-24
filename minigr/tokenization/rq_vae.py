import os
import time
import json
import numpy as np

import torch
from torch.types import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .tokenizer import AbstractTokenizer

from minigr.gr_models import RQVAE
from minigr.utils import log
from minigr.scheduler import get_constant_schedule_with_warmup


class RQVAETokenizer(AbstractTokenizer):
    def __init__(
        self,
        id2item: dict,
        num_codebooks: int,
        codebook_size: int,
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
        
        self.tokenizer = RQVAE(
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
        self.optimizer = torch.optim.AdamW(self.tokenizer.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps)
        log(self.tokenizer)
        
        # Hyperparameters of trainer
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

    def update_metrics(self, loss):
        self.loss = loss
    
    def train_epoch(self, train_loader: DataLoader):
        self.tokenizer.train()
        
        total_loss, total_loss_recon, total_loss_quant = 0, 0, 0
        for batch_data in train_loader:
            self.optimizer.zero_grad()
            x_recon, loss_rq, _ = self.tokenizer(batch_data)
            loss, loss_recon, loss_quant = self.tokenizer.compute_loss(x_recon, batch_data, loss_rq)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_loss_recon += loss_recon.item()
            total_loss_quant += loss_quant.item()
        return total_loss, total_loss_recon, total_loss_quant
    
    def eval_epoch(self, train_embd: Tensor, val_embd: Tensor):
        indices = self.tokenizer.get_indices(train_embd)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        indices_uniq = np.unique(indices, axis=0)
        collision_rate = 1 - (len(indices_uniq) / len(indices))
        utilization = self.tokenizer.get_utilization()
        
        with torch.no_grad():
            val_embd_recon, loss_rq, _ = self.tokenizer(val_embd)
            total_val_loss, _, _ = self.tokenizer.compute_loss(val_embd_recon, val_embd, loss_rq)
        
        self.update_metrics(total_val_loss.item() + collision_rate)
        return indices, collision_rate, utilization, total_val_loss
    
    def _generate_semantic_id(self, sent_embd: Tensor, device: str) -> dict:
        writer = SummaryWriter()
        
        sent_embd = torch.from_numpy(sent_embd).to(device)
        self.tokenizer = self.tokenizer.to(device)
        
        # Notice: the dataset splits (training/valid) are just used to evaluation the quality of codebooks. 
        # We will generate SIDs of all items for downstream generative recommandation tasks
        val_split = int(sent_embd.size(0) * self.val_split)
        indices_shuffled = torch.randperm(sent_embd.size(0))
        val_indices, train_indices = indices_shuffled[:val_split], indices_shuffled[val_split:]
        val_embd, train_embd = sent_embd[val_indices, :], sent_embd[train_indices, :]
        log(f'Train size: {len(train_embd)}, Val size: {len(val_embd)}')
    
        train_loader = DataLoader(train_embd, self.batch_size, shuffle=True)
        val_loader = DataLoader(val_embd, self.batch_size, shuffle=False)
    
        # Initialize the codebooks    
        # for batch_data in train_loader:
        log('Initializing the codebooks...')
        self.tokenizer.init_codebook(train_embd)
        
        best_loss, PATIENCE = float('inf'), 0
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss, loss_recon, loss_quant = self.train_epoch(train_loader)

            indices, collision_rate, utilization, val_loss = self.eval_epoch(train_embd, val_embd)
            if train_loss < best_loss:
                best_loss = train_loss
                PATIENCE = 0
                
                item2sem_ids = {}
                for i in range(indices.shape[0]):
                    item = self.id2item[i + 1]
                    item2sem_ids[item] = tuple(indices[i].tolist())
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
            if PATIENCE >= 200:
                break
            
            self.scheduler.step()
        writer.close()
        return item2sem_ids
    
    def collision_mitigation(self):
        pass
    
    def emb_to_sid(self, sent_embd: Tensor, device: str) -> dict:
        """
        Initialize the tokenizer.

        Args:
            dataset (AmazonReviewsDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        sem_ids_path = os.path.join(self.output_path, f'sentence-t5-base_{self.index_factory}.sem_ids')

        if not os.path.exists(sem_ids_path):
            item2sem_ids = self._generate_semantic_id(sent_embd, device)
            log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
            # with open(sem_ids_path, 'w') as f:
            #     json.dump(item2sem_ids, f)
        else:
            log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
            item2sem_ids = json.load(open(sem_ids_path, 'r'))

        # TODO: 这里需要添加一个防止SID冲突的模块：包含添加第四位和随机改变最后一位
        # item2tokens = self.collision_mitigation(item2sem_ids)
        
        item2tokens = self._sid_shift(item2sem_ids)
        return item2tokens
    