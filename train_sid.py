import argparse

import torch

from nanogr.tokenization import RQVAETokenizer
from nanogr.dataset import AmazonReviews2014
from nanogr.encoder import encode_sent_to_emb
from nanogr.utils import seed_everything


def train():
    seed_everything(args.seed)
    
    raw_dataset = AmazonReviews2014(args.data_path, args.category, args.metadata)
    
    sent_emb = encode_sent_to_emb(raw_dataset, args.projector_path, args.proj_batch_size, device)
    
    # tokenizer = OPQTokenizer(
    #     output_path=args.output_path,
    #     num_codebooks=args.num_codebooks, 
    #     codebook_size=args.codebook_size, 
    #     max_item_seq_len=args.max_item_seq_len,
    #     dataset=raw_dataset
    # )
    
    tokenizer = RQVAETokenizer(
        id2item=raw_dataset.id_mapping['id2item'],
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        hid_channels=args.hid_channels,
        mlp_channels=args.mlp_channels,
        dropout=args.dropout,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        enable_kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        num_warmup_steps=args.num_warmup_steps,
        eval_step=args.eval_step,
        batch_size=args.train_batch_size,
        category=args.category,
        device=device,
    )
    item2tokens = tokenizer.emb_to_sid(sent_emb)
    
    metrics = None
    # metrics = tokenizer.evaluate()
    return metrics
    

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # Shared parameters
    parse.add_argument('--gpu', type=int, default=0)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--data_path', type=str, default='/home/hzzhang/private_data/AmazonReviews2014')
    parse.add_argument('--output_path', type=str, default='./outputs')
    parse.add_argument('--projector_path', type=str, default='/home/hzzhang/Checkpoints/sentence-t5-base')
    
    # Parameters of raw datasets
    parse.add_argument('--category', type=str, default='Toys_and_Games') # [Toys_and_Games, CDs_and_Vinyl]
    parse.add_argument('--metadata', type=str, default='sentence')
    parse.add_argument('--split_strategy', type=str, default='leave_one_out')
    
    # Parameters of the projector
    parse.add_argument('--lr', type=float, default=0.00001)
    parse.add_argument('--wd', type=float, default=0.0)
    parse.add_argument('--epochs', type=float, default=5000)
    parse.add_argument('--eval_step', type=float, default=100)
    parse.add_argument('--num_warmup_steps', type=int, default=1000)
    parse.add_argument('--train_batch_size', type=float, default=99999)
    parse.add_argument('--eval_batch_size', type=float, default=4096)
    parse.add_argument('--proj_batch_size', type=int, default=512)
    
    # Parameters of the tokenizer
    parse.add_argument('--num_codebooks', type=int, default=3)
    parse.add_argument('--codebook_size', type=int, default=256)
    parse.add_argument('--hid_channels', type=int, default=128)
    parse.add_argument('--mlp_channels', type=int, nargs='+', default=[768, 256, 256, 128])
    parse.add_argument('--dropout', type=float, default=0.5)
    parse.add_argument('--quant_loss_weight', type=float, default=0.5)
    parse.add_argument('--kmeans_init', type=bool, default=True)
    parse.add_argument('--kmeans_iters', type=int, default=100)
    parse.add_argument('--beta', type=float, default=0.25)
    args = parse.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train()