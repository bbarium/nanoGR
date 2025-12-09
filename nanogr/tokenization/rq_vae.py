import os
import time
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from .tokenizer import AbstractTokenizer

from nanogr.gr_models import RQVAE
from nanogr.dataset import AmazonReviewsDataset
from nanogr.utils import log
from nanogr.scheduler import get_constant_schedule_with_warmup


class RQVAETokenizer(AbstractTokenizer):
    def __init__(
        self,
        hid_channels: int,
        num_codebooks: int,
        codebook_size: int,
        mlp_channels: list,
        dropout: float,
        quant_loss_weight: float,
        beta: float,
        kmeans_iters: int,
        lr: float,
        wd: float,
        epochs: int,
        num_warmup_steps: int,
        eval_step: int,
        batch_size: int,
        max_item_seq_len: int,
        output_path: str,
        dataset: AmazonReviewsDataset=None,
        device: torch.device=None,
    ):
        super(RQVAETokenizer, self).__init__(num_codebooks, codebook_size, max_item_seq_len, output_path, dataset)
        
        self.tokenizer = RQVAE(
            hid_channels,
            num_codebooks,
            codebook_size,
            mlp_channels=mlp_channels,
            dropout=dropout,
            quant_loss_weight=quant_loss_weight,
            kmeans_iters=kmeans_iters,
            beta=beta,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.tokenizer.parameters(), lr, weight_decay=wd)
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps)
        log(self.tokenizer)
        
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_step = eval_step
        self.hid_channels = hid_channels

    @property
    def index_factory(self) -> str:
        return f'RQVAE{self.num_codebooks}x{self.n_codebook_bits}.dim{self.hid_channels}'

    def _train_epoch(self, train_loader):
        self.tokenizer.train()
        
        total_loss = 0
        for batch_data in train_loader:
            self.optimizer.zero_grad()
            batch_data = batch_data.to(self.device)
            out, rq_loss, _ = self.tokenizer(batch_data)
            loss = self.tokenizer.compute_loss(out, rq_loss, batch_data)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), 1.0) # Gradient Clip
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss
    
    def _generate_semantic_id(self, sent_embs):
        # writer = SummaryWriter()
        dataloader = DataLoader(sent_embs, self.batch_size, shuffle=True)
        
        best_loss, best_collision_rate = 99999, 99999
        for epoch in range(1, self.epochs+1):
            start_time = time.time()
            train_loss = self._train_epoch(dataloader)
            self.scheduler.step(epoch)

            with torch.no_grad():
                embs = torch.from_numpy(sent_embs).to(self.device)
                indices = self.tokenizer.get_indices(embs)
                indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                indices_uniq = np.unique(indices, axis=0)
                collision_rate = 1 - (len(indices_uniq) / len(indices))
                
            if train_loss < best_loss or collision_rate < best_collision_rate:
                best_loss = train_loss
                best_collision_rate = collision_rate
                
                item2sem_ids = {}
                for i in range(indices.shape[0]):
                    item = self.id2item[i + 1]
                    item2sem_ids[item] = tuple(indices[i].tolist())
            end_time = time.time()
            # writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
            # writer.add_scalar('Train/Collision', collision_rate, global_step=epoch)
            
            if epoch == 1 or epoch % self.eval_step == 0:
                log(
                    f'Epoch {epoch}/{self.epochs}, Loss: {train_loss:.4f}, Best Loss: {best_loss:.4f}, ' \
                    f'Best Collision Rate: {best_collision_rate:.2%}, Elapse Time (s): {end_time-start_time:.2f}'
                )
        # writer.close()
        return item2sem_ids
    
    def emb_to_sid(self, sent_embs) -> dict:
        """
        Initialize the tokenizer.

        Args:
            dataset (AmazonReviewsDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        sem_ids_path = os.path.join(self.output_path, f'sentence-t5-base_{self.index_factory}.sem_ids')

        if not os.path.exists(sem_ids_path):
            item2sem_ids = self._generate_semantic_id(sent_embs)
            log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
            with open(sem_ids_path, 'w') as f:
                json.dump(item2sem_ids, f)
        else:
            log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
            item2sem_ids = json.load(open(sem_ids_path, 'r'))

        # TODO: 这里需要添加一个防止SID冲突的模块：包含添加第四位和随机改变最后一位
        
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)
        return item2tokens
    