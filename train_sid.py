import argparse

import torch

from hyperopt import tpe, hp, fmin

from minigr.tokenization import RQVAETokenizer
from minigr.dataset import AmazonReviews2014
from minigr.encoder import encode_sent_to_emb
from minigr.utils import seed_everything, create_log_file, log


def train(params: dict=None):
    if params:
        for key in params.keys():
            if key in args.__dict__.keys():
                args.__dict__[key] = params[key]
    log(args)
    
    seed_everything(args.seed)
    
    # Load raw dataset
    raw_dataset = AmazonReviews2014(args.data_path, args.category, args.metadata)
    
    # Generate item embeddings from metadata
    sent_embd = encode_sent_to_emb(raw_dataset, args.projector_path, args.proj_batch_size, device)
    sent_embd = torch.from_numpy(sent_embd)
    
    tokenizer = RQVAETokenizer(
        id2item=raw_dataset.id_mapping['id2item'],
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        in_channels=sent_embd.size(1),
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
    )
    item2sid = tokenizer.emb_to_sid(sent_embd, device)
    print(item2sid)
    return item2sid


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # Shared parameters
    parse.add_argument('--gpu', type=int, default=1)
    parse.add_argument('--seed', type=int, default=2024)
    parse.add_argument('--data_path', type=str, default='/home/hzzhang/public_data/amazon_data/AmazonReviews2014')
    parse.add_argument('--output_path', type=str, default='./outputs')
    parse.add_argument('--projector_path', type=str, default='/home/hzzhang/Checkpoints/sentence-t5-base')
    
    # Parameters of raw datasets
    parse.add_argument('--category', type=str, default='CDs_and_Vinyl') # [Toys_and_Games, CDs_and_Vinyl]
    parse.add_argument('--metadata', type=str, default='sentence')
    parse.add_argument('--split_strategy', type=str, default='leave_one_out')
    
    # Parameters of the projector
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--wd', type=float, default=0.0)
    parse.add_argument('--epochs', type=float, default=5000)
    parse.add_argument('--eval_step', type=float, default=10)
    parse.add_argument('--num_warmup_steps', type=int, default=50)
    parse.add_argument('--train_batch_size', type=float, default=2048)
    parse.add_argument('--proj_batch_size', type=int, default=512)
    
    # Parameters of the tokenizer
    parse.add_argument('--num_codebooks', type=int, default=3)
    parse.add_argument('--codebook_size', type=int, default=256)
    parse.add_argument('--hid_channels', type=int, default=32)
    parse.add_argument('--mlp_channels', type=int, nargs='+', default=[256, 128, 64])
    parse.add_argument('--kmeans_init', type=bool, default=True)
    parse.add_argument('--kmeans_iters', type=int, default=300)
    parse.add_argument('--dropout', type=float, default=0.0)
    parse.add_argument('--quant_loss_weight', type=float, default=1.0)
    parse.add_argument('--beta', type=float, default=0.25)
    args = parse.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    create_log_file("/home/hzzhang/nanoGR/logs", 'test_CDs_and_Vinyl.log')
    train()
    
    # space = {
    #     'lr': hp.choice('lr', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]),
    #     'wd': hp.choice('wd', [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
    #     'dropout': hp.choice('dropout', [0.0, 0.2, 0.4, 0.6]),
    #     'quant_loss_weight': hp.choice('quant_loss_weight', [0.2, 0.4, 0.6, 0.8, 1.0]),
    #     'beta': hp.choice('beta', [0.25, 0.5, 0.75, 1.0]),
    # }
    # best = fmin(
    #     fn=train,
    #     space=space,
    #     algo=tpe.suggest,
    #     max_evals=50
    # )
    # print(best)