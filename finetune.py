import os
os.environ['WANDB_API_KEY'] = 'ca085b294e483aca15130489d6fa1ed888b3a461'
import json
import argparse
from copy import deepcopy
from collections import defaultdict

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback
)

from nanogr.dataset import AmazonReviews2014
from nanogr.dataloader import textualize
from nanogr.utils import seed_everything, log


def collision_mitigation(item2sid: dict, codebook_size: int) -> dict:
        new_item2sid = deepcopy(item2sid)
        
        sid_counts = defaultdict(int)
        for key, value in new_item2sid.items():
            tmp = '_'.join(map(str, value))
            new_item2sid[key] = value + [sid_counts[tmp]]
            sid_counts[tmp] += 1
            
        max_collision = max(sid_counts.values())
        if max_collision > codebook_size:
            raise ValueError(f'The maximum number of conflicting SIDs ({max_collision}) is greater that the pre-defined codebook size ({codebook_size}).')
        
        num_codebooks = len(next(iter(new_item2sid.values())))
        for item in new_item2sid.keys():
            tokens = list(new_item2sid[item])
            for digit in range(num_codebooks):
                tokens[digit] += codebook_size * digit + 1
            new_item2sid[item] = list(map(lambda t: f'<{chr(97+digit)}_{t}>', tokens))
        return new_item2sid


class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch_data):
        input_text = [data['input_ids'] for data in batch_data]
        label_text = [data['label_ids'] for data in batch_data]
        
        tokenized_input = self.tokenizer(
            input_text, 
            return_tensors='pt', 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True
        )
        tokenized_label = self.tokenizer(
            label_text, 
            return_tensors='pt', 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True
        )
        
        tokenized_input['labels'] = tokenized_label['input_ids']
        tokenized_input['labels'][tokenized_input['labels'] == self.tokenizer.pad_token_id] = -100
        return tokenized_input


def gr_pipeline():
    seed_everything(args.seed)
    
    # TODO: 添加多模态数据集
    # Load Raw Dataset
    raw_dataset = AmazonReviews2014(args.data_path, args.category, args.metadata)
    split_datasets = raw_dataset.split(args.split_strategy)
    
    max_length = 0
    for item_seq in split_datasets['train']['item_seq']:
        max_length = max(len(item_seq), max_length)
    log(f'[DATASET] Max Length of item sequences: {max_length}')
    
    log(f'[TOKENIZER] Loading semantic IDs from {args.sem_ids_path}...')
    item2sid = json.load(open(args.sem_ids_path, 'r'))
    item2sid = collision_mitigation(item2sid, 256)
       
    # Instantiate a tokenization and add new SIDs
    log(f'[TOKENIZER] Add new SIDs to the Tokenization...')
    tokenizer = AutoTokenizer.from_pretrained(args.gr_path)
    new_tokens = []
    for sid in item2sid.values():
        new_tokens += sid
    new_tokens = set(new_tokens)
    tokenizer.add_tokens(list(new_tokens))
    new_vocab_size = len(tokenizer)
    tokenizer.model_max_length = args.model_max_length
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    # tokenizer.model_max_length = args.model_max_length + max_length*
    log(f'[TOKENIZER] New Vocabulary Size: {new_vocab_size}')
    
    # Instantiate an LLM
    if args.train_from_scratch:
        config = AutoConfig.from_pretrained(args.gr_path)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.gr_path, 
            device_map='auto',
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    model.resize_token_embeddings(new_vocab_size)
    
    # Create train/val splits
    if not os.path.exists(args.tmp_path):
        tokenized_datasets = textualize(split_datasets, item2sid)
        torch.save(tokenized_datasets, './tokenized_datasets.pt')
    else:
        tokenized_datasets = torch.load(args.tmp_path, weights_only=False)
    
    # Train and Eval Steps
    collator = Collator(tokenizer)
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        args=TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_step,
            optim=args.optim,
            # batch_eval_metrics=True,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_step,
            save_steps=args.save_and_eval_step,
            output_dir=args.output_path,
            save_total_limit=2,
            load_best_model_at_end=True,
            eval_delay= 1,
            # report_to='none'
        ),
        # compute_metrics=calculate_metrics,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=40)]
    )
    
    trainer.train()
    return


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # Shared parameters
    parse.add_argument('--gpu', type=int, default=0)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--per_device_batch_size', type=int, default=64)
    parse.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parse.add_argument('--warmup_ratio', type=float, default=0.01)
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--learning_rate', type=float, default=5e-4)
    parse.add_argument('--weight_decay', type=float, default=0.01)
    parse.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parse.add_argument('--optim', type=str, default='adamw_torch')
    parse.add_argument('--save_and_eval_strategy', type=str, default='epoch')
    parse.add_argument('--save_and_eval_step', type=int, default=50)
    parse.add_argument('--logging_step', type=int, default=20)
    
    # Parameters of local file paths
    parse.add_argument('--data_path', type=str, default='/home/hzzhang/public_data/amazon_data/AmazonReviews2014')
    parse.add_argument('--sem_ids_path', type=str, default='/home/hzzhang/nanoGR/outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids')
    parse.add_argument('--gr_path', type=str, default='/home/hzzhang/Checkpoints/Qwen3-0.6B')
    parse.add_argument('--output_path', type=str, default='./outputs')
    parse.add_argument('--tmp_path', type=str, default='./tokenized_datasets.pt')
    
    # Parameters of raw datasets
    parse.add_argument('--category', type=str, default='Toys_and_Games') # [Toys_and_Games, CDs_and_Vinyl]
    parse.add_argument('--metadata', type=str, default='sentence')
    parse.add_argument('--split_strategy', type=str, default='leave_one_out')
    
    # Parameters of the LLM
    parse.add_argument('--train_from_scratch', type=bool, default=False)
    parse.add_argument('--model_max_length', type=int, default=128)
    args = parse.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    gr_pipeline()