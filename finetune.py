import os
import json
import argparse
import datetime
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
from nanogr.dataloader import tokenize, create_hf_datasets
from nanogr.utils import seed_everything, log, NEW_TOKENS


def collision_mitigation(item2sid: dict, codebook_size: int=256) -> dict:
    new_item2sid = deepcopy(item2sid)
    
    sid_counts = defaultdict(int)
    for key, value in new_item2sid.items():
        tmp = '_'.join(map(str, value))
        new_item2sid[key] = value + [sid_counts[tmp]]
        sid_counts[tmp] += 1
        
    max_collision = max(sid_counts.values())
    if max_collision > codebook_size:
        raise ValueError(f'The maximum number of conflicting SIDs ({max_collision}) is greater that the pre-defined codebook size ({codebook_size}).')
    
    for item in new_item2sid.keys():
        token2str = []
        for digit in range(len(new_item2sid[item])):
            sid = new_item2sid[item][digit]
            sid = sid + codebook_size * digit + 1
            token2str.append(f'<{chr(97+digit)}_{sid}>')
        new_item2sid[item] = token2str
    return new_item2sid


def extend_tokenization_(tokenizer: AutoTokenizer, item2sid: dict):
    new_tokens = []
    for sid in item2sid.values():
        new_tokens += sid
    new_tokens = sorted(set(new_tokens))  # sort to ensure deterministic token ID assignment across runs
    tokenizer.add_tokens(new_tokens + NEW_TOKENS) # add all sids and new special token ('<|sid_start|>', '<|sid_end|>', '<|rec|>') into the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"


def gr_pipeline():
    seed_everything(args.seed)
    
    # TODO: 添加多模态数据集
    # Load Raw Dataset
    raw_dataset = AmazonReviews2014(args.data_path, args.category, args.metadata)
    split_datasets = raw_dataset.split(args.split_strategy)
    max_len, min_len = raw_dataset.max_min_seq_len
    log(f'[DATASET] Maximum/Minimum lengths of item sequences: {max_len}, {min_len}')
    
    log(f'[TOKENIZER] Loading semantic IDs from {args.sem_ids_path}...')
    item2sid = json.load(open(args.sem_ids_path, 'r'))
    item2sid = collision_mitigation(item2sid)
       
    # Instantiate a tokenization and add new SIDs
    log(f'[TOKENIZER] Add new SIDs to the Tokenization...')
    tokenizer = AutoTokenizer.from_pretrained(args.gr_path)
    old_vocab_size = len(tokenizer)
    extend_tokenization_(tokenizer, item2sid)
    new_vocab_size = len(tokenizer)
    log(f'[TOKENIZER] New vocabulary size: {new_vocab_size}, the number of new tokens: {new_vocab_size-old_vocab_size}')
    
    # Create train/val/test splits
    # TODO: 不应该从本地加载tokenized dataset，这可能会导致模型误加载基于上一个tokenizer生成的数据集，此时数据集里的token id和tokenizer的token id是不匹配的
    max_seq_length = min(args.max_seq_length, max_len)
    if not os.path.exists(args.tmp_path):
        tokenized_datasets = tokenize(split_datasets, item2sid, tokenizer, max_seq_length)
        torch.save(tokenized_datasets, f'./{args.category}.tokenized_datasets.pt')
    else:
        tokenized_datasets = torch.load(args.tmp_path, weights_only=False)
    hf_datasets = create_hf_datasets(tokenized_datasets)

    # Instantiate an LLM
    if args.train_from_scratch:
        config = AutoConfig.from_pretrained(args.gr_path)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.gr_path, 
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    model.resize_token_embeddings(new_vocab_size)

    if args.run_name is None:
        args.run_name = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M")}'
    
    # Train and Eval Steps
    train_args = TrainingArguments(
        output_dir=args.output_path,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        bf16=True, # not suitable for P40
        tf32=True, # not suitable for P40
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=args.logging_step,
        eval_strategy=args.save_and_eval_strategy,
        eval_steps=args.save_and_eval_step,
        save_strategy=args.save_and_eval_strategy,
        save_steps=args.save_and_eval_step,
        save_total_limit=2,
        load_best_model_at_end=(args.save_and_eval_strategy != "no"),
        report_to=['tensorboard'],
        run_name=args.run_name,
        project=args.project,
        disable_tqdm=False,
        ddp_find_unused_parameters=False, # 多卡训练常见优化
        torch_compile=args.enable_torch_compile
    )

    trainer = Trainer(
        model=model,
        train_dataset=hf_datasets['train'],
        eval_dataset=hf_datasets['val'],
        args=train_args,
        processing_class=tokenizer,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop)]
    )
    trainer.train()
    return


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # Parameters of local file paths
    parse.add_argument('--data_path', type=str, default='./AmazonReviews')
    parse.add_argument('--gr_path', type=str, default='./Checkpoints/Qwen3.5-0.8B')
    parse.add_argument('--sem_ids_path', type=str, default='./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids')
    parse.add_argument('--output_path', type=str, default='./outputs')
    parse.add_argument('--log_path', type=str, default='./logs')
    
    # Parameters of raw datasets
    parse.add_argument('--category', type=str, default='Toys_and_Games') # [Toys_and_Games, CDs_and_Vinyl]
    parse.add_argument('--metadata', type=str, default='sentence')
    parse.add_argument('--split_strategy', type=str, default='leave_one_out')
    
    # Parameters of the LLM
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--per_device_batch_size', type=int, default=60)
    parse.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parse.add_argument('--epochs', type=int, default=500)
    parse.add_argument('--save_and_eval_strategy', type=str, default='epoch')
    parse.add_argument('--save_and_eval_step', type=int, default=50)
    parse.add_argument('--logging_step', type=int, default=20)
    parse.add_argument('--project', type=str, default='nanoGR-sft')
    parse.add_argument('--run_name', type=str, default=None)
    parse.add_argument('--optim', type=str, default='adamw_torch')
    parse.add_argument('--learning_rate', type=float, default=5e-4)
    parse.add_argument('--weight_decay', type=float, default=0.01)
    parse.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parse.add_argument('--warmup_ratio', type=float, default=0.01)
    parse.add_argument('--early_stop', type=int, default=100)
    
    parse.add_argument("--dataloader_num_workers", type=int, default=8)
    parse.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parse.add_argument('--max_seq_length', type=int, default=30)
    parse.add_argument('--train_from_scratch', action='store_true')
    parse.add_argument("--enable_torch_compile", action="store_true")
    args = parse.parse_args()
    
    gr_pipeline()