import argparse
import datetime
import json
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from nanogr.dataset import AmazonReviews2014
from nanogr.nn import SemanticIDEncoderDecoder
from nanogr.decoder import init_decoder
from nanogr.encoder import init_encoder
from nanogr.trainer import Trainer
from nanogr.dist_dataloader import build_or_load_datasets, create_dataloaders
from nanogr.utils import (
    Config,
    create_log_file,
    log,
    seed_everything,
    collision_mitigation,
)


def _setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def _is_main(rank: int) -> bool:
    return rank == 0


def build_trainer_config(args, train_loader: DataLoader, is_distributed: bool, local_rank: int) -> Config:
    config_dict = vars(args).copy()
    if torch.cuda.is_available():
        config_dict["device"] = torch.device(f"cuda:{local_rank}")
    else:
        config_dict["device"] = torch.device("cpu")
    config_dict["eval_topks"] = sorted({int(k) for k in args.eval_topks.split(",") if k})
    config_dict["metric_for_best_model"] = f"hit@{5 if 5 in config_dict['eval_topks'] else config_dict['eval_topks'][0]}"
    total_update_steps = max(
        (len(train_loader) * args.epochs) // max(args.gradient_accumulation_steps, 1),
        1,
    )
    config_dict["num_warmup_steps"] = int(total_update_steps * args.warmup_ratio)
    config_dict["output_dir"] = os.path.join(args.output_path, args.run_name)
    config_dict["is_distributed"] = is_distributed
    config_dict["local_rank"] = local_rank
    return Config(config_dict)


def gr_pipeline(args):
    is_distributed, rank, world_size, local_rank = _setup_distributed()

    seed_everything(args.seed)

    if args.run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M") if _is_main(rank) else ""
        if is_distributed:
            obj_list = [run_name]
            dist.broadcast_object_list(obj_list, src=0)
            run_name = obj_list[0]
        args.run_name = run_name

    if _is_main(rank):
        create_log_file(args.log_path, f"{args.run_name}.log")
        log(args)

    # Load the raw dataset from local and create sequence splits controled by the maximum sequence length.
    raw_dataset = AmazonReviews2014(args.data_path, args.category, args.metadata)
    max_raw_len, min_raw_len = raw_dataset.max_min_seq_len
    if _is_main(rank):
        log(f"[DATASET] Maximum/Minimum lengths of item sequences: {max_raw_len}, {min_raw_len}")

    max_seq_len = min(args.max_seq_length, max_raw_len) if max_raw_len > 0 else args.max_seq_length
    split_datasets = raw_dataset.split(args.split_strategy, max_seq_len, training_window=args.training_window)
    if _is_main(rank):
        log(
            f"[DATASET] Train/Val/Test splits via {args.split_strategy} (training_window={args.training_window}) \n"
            f"#train={len(split_datasets['train']['user'])}, "
            f"#val={len(split_datasets['val']['user'])}, "
            f"#test={len(split_datasets['test']['user'])}"
        )

    if _is_main(rank):
        log(f"[TOKENIZER] Loading semantic IDs from {args.sem_ids_path}...")
    with open(args.sem_ids_path, "r") as file:
        item2sid = json.load(file)
    item2sid = collision_mitigation(item2sid, args.codebook_size) # Generate an extra SID for overcoming the SID collision
    codebooks = torch.tensor(list(item2sid.values()), dtype=torch.long)
    num_codebooks = len(next(iter(item2sid.values())))

    if _is_main(rank):
        log("[DATALOADER] Create Train/Val/Test dataloaders...")
    sid_seq_datasets = build_or_load_datasets(split_datasets, item2sid, 
        category=args.category, num_codebooks=num_codebooks, max_seq_length=args.max_seq_length, 
        output_path=args.output_path, is_distributed=is_distributed, rank=rank
    )
    dataloaders = create_dataloaders(sid_seq_datasets, num_workers=args.dataloader_num_workers,
        per_device_batch_size=args.per_device_batch_size, eval_batch_size=args.eval_batch, 
        prefetch_factor=args.dataloader_prefetch_factor, is_distributed=is_distributed, rank=rank, world_size=world_size
    )
    trainer_config = build_trainer_config(args, dataloaders[0], is_distributed, local_rank)

    total_vocab_size = num_codebooks * args.codebook_size + 1
    cfg = Config.load_config_from_file(args.config_path) # Load encoder/decoder hyperparameters from YAML configs
    encoder_cfg, decoder_cfg = cfg.encoder, cfg.decoder
    encoder_cfg["vocab_size"] = decoder_cfg["vocab_size"] = total_vocab_size

    encoder = init_encoder(encoder_cfg.backbone, encoder_cfg)
    decoder = init_decoder(decoder_cfg.backbone, decoder_cfg)
    model = SemanticIDEncoderDecoder(
        num_hierarchies=num_codebooks,
        num_embeddings_per_hierarchy=args.codebook_size,
        codebooks=codebooks,
        encoder=encoder,
        decoder=decoder,
        d_model=encoder_cfg.d_model,
        num_beams=max(args.num_beams, max(trainer_config.eval_topks)),
        loss_func=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=-100),
        check_prefix=args.enable_check_prefix,
    )
    if _is_main(rank):
        log(model)

    if args.enable_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    trainer = Trainer(trainer_config, model)
    metrics = trainer.fit(dataloaders)
    if _is_main(rank):
        log(f"[TEST] Final metrics: {metrics}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="./data/AmazonReviews")
    parser.add_argument("--sem_ids_path", type=str, default="./outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--config_path", type=str, default="./configs/enc_dec_gr_plain.yaml")
    parser.add_argument("--project", type=str, default="nanoGR")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--category", type=str, default="Toys_and_Games")
    parser.add_argument("--metadata", type=str, default="sentence")
    parser.add_argument("--split_strategy", type=str, default="leave_one_out")
    parser.add_argument("--training_window", type=str, default="sliding", choices=["sliding", "last_only"])
    parser.add_argument("--rebuild_dataset_cache", action="store_true")

    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parser.add_argument("--per_device_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_and_eval_step", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--model_arch", type=str, default="enc_dec", choices=["enc_dec", "dec_only"])
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=20)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--eval_topks", type=str, default="5,10")
    parser.add_argument("--enable_check_prefix", action="store_true")
    parser.add_argument("--enable_torch_compile", action="store_true")
    args = parser.parse_args()

    gr_pipeline(args)