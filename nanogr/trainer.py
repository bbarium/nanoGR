from typing import Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer

from nanogr.utils import Config
from nanogr.scheduler import get_constant_schedule_with_warmup


class Trainer:
    def __init__(self, trainer_configs: dict, model: nn.Module, tokenizer: AutoTokenizer):
        self.configs = trainer_configs
        self.model = model
        self.tokenizer = tokenizer
        
    def save_model_ckpt():
        pass
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, phase: str='val'):
        self.model.eval()
        for batch_data in tqdm(dataloader, total=len(dataloader), desc=f'{phase.upper()} Phase: '):
            self.model(batch_data)
  
    def train(self, train_dataloader: DataLoader):
        self.model.train()
        
        for batch_data in tqdm(train_dataloader, total=len(train_dataloader), desc='TRAIN Phase: '):
            input_text = batch_data['input_ids']
            label_text = batch_data['label_ids']
            
            tokenized_input = self.tokenizer(
                input_text, 
                return_tensors='pt', 
                padding='longest', 
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True
            )
            tokenized_label = self.tokenizer(
                label_text, 
                return_tensors='pt', 
                padding='longest', 
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True
            )
            
            tokenized_input['labels'] = tokenized_label['input_ids']
            tokenized_input['labels'][tokenized_input['labels'] == self.tokenizer.pad_token_id] = -100
            embd = self.model(tokenized_input)
            
            self.optimizer.zero_grad()
    
    def _align_special_tokens():
        pass
    
    def fit(self, dataloaders: Tuple[DataLoader, DataLoader, DataLoader]):
        train_dataloader, val_dataloader, test_dataloader = dataloaders
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.configs.lr, weight_decay=self.configs.wd)
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, self.configs.num_warmup_steps)
        
        best_hit_5 = 0
        for epoch in range(1, self.configs.epochs+1):
            self.train(train_dataloader)
            transformers.Trainer()
            
            if epoch == 1 or epoch % self.configs.eval_step == 0:
                metrics = self.evaluate(val_dataloader)
                
                if metrics[0] > best_hit_5:
                    best_hit_5 = metrics[0]
                    self.save_model_ckpt()
        
        metrics = self.evaluate(test_dataloader)
        return metrics
        