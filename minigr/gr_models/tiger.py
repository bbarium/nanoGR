from typing import Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.types import Tensor

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM, 
    T5Config, 
    T5ForConditionalGeneration
)
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import Seq2SeqModelOutput

from minigr.utils import delete_module


class T5Encoder(nn.Module):
    def __init__(self, encoder: transformers.PreTrainedModel, num_embd: int, embd_dim: int):
        super().__init__()
        self.encoder = encoder
        self.num_embd = num_embd
        self.embd_dim = embd_dim
        
        delete_module(self.encoder, "embed_tokens")
        delete_module(self.encoder, "shared")
    
        self.reset_parameters(self.encoder)
    
    def reset_parameters(self, module: nn.Module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        
        for submodule in module.children():
            self.reset_parameters(submodule)
            
    def forward(self, seq_embd: Tensor, attn_mask: Tensor) -> Tensor:
        output = self.encoder(inputs_embeds=seq_embd, attention_mask=attn_mask)
        return output.last_hidden_state
    

class T5Decoder(nn.Module):
    def __init__(
        self, 
        decoder: transformers.PreTrainedModel, 
        decoder_mlp: nn.Module=None, 
        bos_token: nn.Parameter=None
    ):
        super().__init__()
        self.decoder = decoder
        self.decoder_mlp = decoder_mlp
        self.bos_token = bos_token
        
        delete_module(self.decoder, "embed_tokens")
        delete_module(self.decoder, "shared")
        self.reset_parameters(self.decoder)

    def reset_parameters(self, module: nn.Module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        
        for submodule in module.children():
            self.reset_parameters(submodule)
    
    def forward(
        self,
        seq_embd: Tensor, 
        attn_mask: Tensor,
        encoder_output: Tensor,
        encoder_attn_mask: Tensor,
        use_cache: bool=False,
        past_key_values: DynamicCache=DynamicCache(),
    ):
        decoder_outputs: Seq2SeqModelOutput = self.decoder(
            attention_mask=attn_mask,
            inputs_embeds=seq_embd,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attn_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        return decoder_outputs.last_hidden_state


class TIGER(nn.Module):
    def __init__(
        self, 
        base_model: Union[str, Path],
        item_id2tokens: dict,
        d_model: int,
        tokenizer=None,
        model=None,
    ):
        super().__init__()
        bos_token = torch.nn.Parameter(torch.randn(1, d_model), requires_grad=True)
        
        # TODO: 这里需要添加大模型加速的方案
        if 't5' in base_model.lower():
            self.model = T5ForConditionalGeneration.from_pretrained(base_model, device_map='auto')
        elif 'qwen' in base_model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForCausalLM.from_pretrained(base_model)
        else:
            raise NotImplementedError
        
        self.item_id2tokens = item_id2tokens
    
    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    @property
    def total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def forward(self, batch: dict, return_loss: bool=True) -> Tensor:
        input_tokens  = self.item_id2tokens[batch['input_ids']]
        labels_tokens = self.item_id2tokens[batch['labels']]
        
        num_seq_tokens = input_tokens.size(1) * input_tokens.size(2)
        input_tokens, labels_tokens = input_tokens.reshape(-1, num_seq_tokens), labels_tokens.reshape(-1, num_seq_tokens)
        outputs = self.model(input_ids=input_tokens, labels=labels_tokens)
        output_embd = outputs.last_hidden_state
        
        if return_loss:
            loss = None
            return output_embd, loss
        
        return output_embd
    
    def compute_loss(self, ) -> Tensor:
        pass