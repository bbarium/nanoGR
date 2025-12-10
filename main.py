import os
import argparse
import torch

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback
)

from nanogr.dataset import AmazonReviews2014

from nanogr.evaluator import Evaluator
from nanogr.encoder import encode_sent_to_emb
from nanogr.utils import seed_everything


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
    
    # Load Raw Dataset
    # TODO: 添加多模态的数据
    raw_dataset = AmazonReviews2014(args.data_path, args.category, args.metadata)
    split_datasets = raw_dataset.split(args.split_strategy)
    
    # Dataset Pre-processing
    # TODO: 需要添加预处理模块用于处理原始文本或者多模态数据
    
    # Text2Emb
    sent_emb = encode_sent_to_emb(raw_dataset, args.projector_path, args.proj_batch_size, device)

    # Emb2SID
    training_item_mask = raw_dataset.training_item_mask()
    
    if os.path.exists('./tokenized_datasets.pt'):
        tokenized_datasets = torch.load('./tokenized_datasets.pt')
        item2tokens = tokenized_datasets.item2tokens()
    # else:
    #     tokenized_datasets = tokenizer.tokenize(split_datasets)
    #     torch.save(tokenized_datasets, 'tokenized_datasets.pt')
    
    # Instantiate LLM-driven Model
    new_tokens = []
    for sid in item2tokens.values():
        new_tokens += sid
    new_tokens = set(new_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.gr_path)
    num_new_tokens = tokenizer.add_tokens(list(new_tokens))
    new_vocab_size = len(tokenizer)
    # logg
    tokenizer.model_max_length = args.model_max_length
    
    model = AutoModelForCausalLM.from_pretrained(args.gr_path, device_map='auto')
    model.resize_token_embeddings(new_vocab_size)
    
    # model = TIGER(args.model_path, item_id2tokens, args.d_model)
    
    # Train and Eval Steps
    # metrics = Trainer(args, model, tokenizer).fit(dataloaders)
    evaluator = Evaluator()
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
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_step,
            save_steps=args.save_and_eval_step,
            output_dir=args.output_path,
            save_total_limit=2,
            load_best_model_at_end=True,
            eval_delay= 1,
            compute_metrics=evaluator.calculate_metrics
        ),
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
    parse.add_argument('--per_device_batch_size', type=int, default=72)
    parse.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parse.add_argument('--warmup_ratio', type=float, default=0.01)
    parse.add_argument('--epochs', type=int, default=200)
    parse.add_argument('--learning_rate', type=float, default=5e-4)
    parse.add_argument('--weight_decay', type=float, default=0.01)
    parse.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parse.add_argument('--optim', type=str, default='adamw_torch')
    parse.add_argument('--save_and_eval_strategy', type=str, default='epoch')
    parse.add_argument('--save_and_eval_step', type=int, default=1000)
    parse.add_argument('--logging_step', type=int, default=10)
    
    # Parameters of local file paths
    parse.add_argument('--data_path', type=str, default='/home/hzzhang/private_data/AmazonReviews2014')
    parse.add_argument('--output_path', type=str, default='./outputs')
    parse.add_argument('--projector_path', type=str, default='/home/hzzhang/Checkpoints/sentence-t5-base')
    parse.add_argument('--gr_path', type=str, default='/home/hzzhang/Checkpoints/Qwen3-0.6B')
    
    # Parameters of raw datasets
    parse.add_argument('--category', type=str, default='Toys_and_Games') # [Toys_and_Games, CDs_and_Vinyl]
    parse.add_argument('--metadata', type=str, default='sentence')
    parse.add_argument('--split_strategy', type=str, default='leave_one_out')
    
    # Parameters of the projector
    parse.add_argument('--lr', type=float, default=0.0001)
    parse.add_argument('--wd', type=float, default=0.0)
    # parse.add_argument('--epochs', type=float, default=5000)
    parse.add_argument('--eval_step', type=float, default=100)
    parse.add_argument('--num_warmup_steps', type=int, default=1000)
    parse.add_argument('--train_batch_size', type=float, default=4096)
    parse.add_argument('--eval_batch_size', type=float, default=4096)
    parse.add_argument('--proj_batch_size', type=int, default=512)
    
    # Parameters of the tokenizer
    parse.add_argument('--num_codebooks', type=int, default=3)
    parse.add_argument('--codebook_size', type=int, default=256)
    parse.add_argument('--max_item_seq_len', type=int, default=50)
    parse.add_argument('--hid_channels', type=int, default=64)
    parse.add_argument('--mlp_channels', type=int, nargs='+', default=[768, 256, 128, 64])
    parse.add_argument('--dropout', type=float, default=0.0)
    parse.add_argument('--quant_loss_weight', type=float, default=1.0)
    parse.add_argument('--kmeans_init', type=bool, default=False)
    parse.add_argument('--kmeans_iters', type=int, default=100)
    parse.add_argument('--beta', type=float, default=0.25)
    
    # Parameters of the LLM
    parse.add_argument('--d_model', type=int, default=512)
    parse.add_argument('--num_layers', type=int, default=6)
    parse.add_argument('--num_heads', type=int, default=8)
    parse.add_argument('--temperature', type=float, default=0.6)
    parse.add_argument('--model_max_length', type=int, default=25)
    args = parse.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    gr_pipeline()