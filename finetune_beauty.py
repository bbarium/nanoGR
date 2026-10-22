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
from nanogr.dataloader import ItemSeqDataset, create_text_dataloader
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


def extend_tokenization_(tokenizer: AutoTokenizer, num_codebooks: int, codebook_size: int):
    new_tokens = []
    for digit in range(num_codebooks):
        new_tokens += \
            [f'<{chr(97+digit)}_{i + codebook_size * digit + 1}>' for i in range(codebook_size)] # Notice: SID starts from 1

    tokenizer.add_tokens(new_tokens + NEW_TOKENS) # add SIDs and new special tokens ('<|sid_start|>', '<|sid_end|>', '<|rec|>') into the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"


class BatchTokenizedTrainer(Trainer):
    def __init__(self, *args, train_dataloader=None, eval_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader

    def get_train_dataloader(self):
        if self._train_dataloader is not None:
            return self._train_dataloader
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is not None and eval_dataset is not self.eval_dataset:
            return super().get_eval_dataloader(eval_dataset)
        if self._eval_dataloader is not None:
            return self._eval_dataloader
        return super().get_eval_dataloader(eval_dataset)


def gr_pipeline():
    seed_everything(args.seed)

    # Load Raw Dataset
    raw_dataset = AmazonReviews2014(args.data_path, args.category, args.metadata)
    max_raw_len, min_raw_len = raw_dataset.max_min_seq_len
    log(f'[DATASET] Maximum/Minimum lengths of item sequences: {max_raw_len}, {min_raw_len}')

    # TODO: For max_seq_len == -1, we should directly use the raw item sequences.
    log(f'[DATASET] Create Train/Val/Test splits via {args.split_strategy}...')
    max_seq_len = min(args.max_seq_length, max_raw_len) # -1 - using raw item sequences
    split_datasets = raw_dataset.split(args.split_strategy, max_seq_len)

    log(f'[TOKENIZER] Loading semantic IDs from {args.sem_ids_path}...')
    with open(args.sem_ids_path, 'r') as file:
        item2sid = json.load(file)
    item2sid = collision_mitigation(item2sid, args.codebook_size)

    # Instantiate a tokenization and add new SIDs
    log(f'[TOKENIZER] Add new SIDs to the Tokenization...')
    tokenizer = AutoTokenizer.from_pretrained(args.gr_path)
    old_vocab_size = len(tokenizer)
    extend_tokenization_(tokenizer, args.num_codebooks+1, args.codebook_size)
    new_vocab_size = len(tokenizer)
    log(f'[TOKENIZER] New vocabulary size: {new_vocab_size}, the number of new tokens: {new_vocab_size-old_vocab_size}')

    train_dataset = ItemSeqDataset(split_datasets['train'], item2sid, split='train')
    val_dataset = ItemSeqDataset(split_datasets['val'], item2sid, split='val')
    train_dataloader = create_text_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )
    val_dataloader = create_text_dataloader(
        dataset=val_dataset,
        tokenizer=tokenizer,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )

    # Instantiate an LLM
    if args.train_from_scratch:
        config = AutoConfig.from_pretrained(args.gr_path)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.gr_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
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
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        bf16=True,
        tf32=True,
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
        disable_tqdm=False,
        ddp_find_unused_parameters=False, # 多卡训练常见优化
        torch_compile=args.enable_torch_compile
    )

    trainer = BatchTokenizedTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=train_args,
        processing_class=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop)]
    )
    trainer.train()
    return


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # Parameters of local file paths
    parse.add_argument('--data_path', type=str, default='./data/AmazonReviews')
    parse.add_argument('--gr_path', type=str, default='./checkpoints/Qwen3-0.6B')
    parse.add_argument('--sem_ids_path', type=str, default='./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids')
    parse.add_argument('--output_path', type=str, default='./outputs')
    parse.add_argument('--log_path', type=str, default='./logs')

    # Parameters of raw datasets
    parse.add_argument('--category', type=str, default='Toys_and_Games') # [Toys_and_Games, CDs_and_Vinyl]
    parse.add_argument('--metadata', type=str, default='sentence')
    parse.add_argument('--split_strategy', type=str, default='leave_one_out')

    # Parameters of the LLM
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--per_device_batch_size', type=int, default=30)
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
    parse.add_argument('--num_codebooks', type=int, default=3)
    parse.add_argument('--codebook_size', type=int, default=256)
    parse.add_argument('--max_seq_length', type=int, default=30)
    parse.add_argument('--train_from_scratch', action='store_true')
    parse.add_argument("--enable_torch_compile", action="store_true")
    parse.add_argument("--max_steps", type=int, default=-1)
    args = parse.parse_args()

    gr_pipeline()
