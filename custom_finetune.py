import argparse
import datetime
import hashlib
import json
import os
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from nanogr.dataset import AmazonReviews2014
from nanogr.nn import SemanticIDEncoderDecoder
from nanogr.decoder import init_decoder
from nanogr.encoder import init_encoder
from nanogr.trainer import Trainer
from nanogr.utils import (
    Config,
    create_log_file,
    log,
    seed_everything,
    collision_mitigation,
)


def _setup_distributed():
    """Read torchrun env vars and initialise the process group.

    Returns a tuple ``(is_distributed, rank, world_size, local_rank)``.  When the script
    is launched without ``torchrun`` we simply return single-process defaults so the
    rest of the pipeline is unaffected.
    """
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


class SemanticIDDataset(Dataset):
    def __init__(
        self,
        split_dataset: dict=None,
        item2sid: Dict[str, List[int]]=None,
        num_hierarchies: int=None,
        max_history_len: int=None,
        from_tensors: Dict[str, torch.Tensor]=None,
    ) -> None:
        if from_tensors is not None:
            self.input_ids = from_tensors["input_ids"]
            self.attention_mask = from_tensors["attention_mask"]
            self.fut_ids = from_tensors["fut_ids"]
            return

        assert split_dataset is not None and item2sid is not None
        max_token_len = max_history_len * num_hierarchies

        item_keys = list(item2sid.keys())
        item_to_idx = {k: i for i, k in enumerate(item_keys)}
        sid_table = np.asarray([item2sid[k] for k in item_keys], dtype=np.int64)
        assert sid_table.shape[1] == num_hierarchies, \
            f"item2sid hierarchies ({sid_table.shape[1]}) != num_hierarchies ({num_hierarchies})"

        item_seqs = split_dataset["item_seq"]
        fut_items = split_dataset["fut_item"]
        n_samples = len(item_seqs)

        input_ids = np.zeros((n_samples, max_token_len), dtype=np.int64)
        attention_mask = np.zeros((n_samples, max_token_len), dtype=np.int64)
        fut_ids = np.zeros((n_samples, num_hierarchies), dtype=np.int64)

        show_pbar = _is_main(int(os.environ.get("RANK", 0)))
        for i, (history, target) in enumerate(
            tqdm(
                zip(item_seqs, fut_items),
                total=n_samples,
                disable=not show_pbar,
                desc="[DATASET] tokenize",
            )
        ):
            history = history[-max_history_len:]
            if history:
                hist_idx = np.fromiter(
                    (item_to_idx[h] for h in history),
                    dtype=np.int64,
                    count=len(history),
                )
                flat_tokens = sid_table[hist_idx].reshape(-1)
                k = flat_tokens.shape[0]
                input_ids[i, max_token_len - k :] = flat_tokens
                attention_mask[i, max_token_len - k :] = 1
            fut_ids[i] = sid_table[item_to_idx[target]]

        self.input_ids = torch.from_numpy(input_ids)
        self.attention_mask = torch.from_numpy(attention_mask)
        self.fut_ids = torch.from_numpy(fut_ids)

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "fut_ids": self.fut_ids[index],
        }


def _dataset_cache_key(args, split_datasets: dict, item2sid: Dict[str, List[int]]) -> str:
    h = hashlib.sha1()
    h.update(
        json.dumps(
            {
                "category": args.category,
                "split_strategy": args.split_strategy,
                "training_window": args.training_window,
                "max_seq_length": args.max_seq_length,
                "num_codebooks": args.num_codebooks,
                "codebook_size": args.codebook_size,
                "sem_ids_path": os.path.abspath(args.sem_ids_path),
            },
            sort_keys=True,
        ).encode()
    )

    # Cheap content fingerprint – size + first/last few items per split.
    for split in ("train", "val", "test"):
        d = split_datasets[split]
        h.update(f"{split}:{len(d['user'])}".encode())
        if d["fut_item"]:
            h.update(
                json.dumps(
                    [d["fut_item"][0], d["fut_item"][-1], d["item_seq"][0][:4], d["item_seq"][-1][:4]]
                ).encode()
            )
    h.update(str(len(item2sid)).encode())
    if item2sid:
        first_key = next(iter(item2sid.keys()))
        h.update(f"{first_key}:{item2sid[first_key]}".encode())
    return h.hexdigest()[:16]


def _build_split_tensors(
    split_datasets: dict,
    item2sid: Dict[str, List[int]],
    num_codebooks: int,
    max_seq_length: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    out = {}
    for split in ("train", "val", "test"):
        sid_seq_dataset = SemanticIDDataset(
            split_dataset=split_datasets[split],
            item2sid=item2sid,
            num_hierarchies=num_codebooks,
            max_history_len=max_seq_length,
        )
        out[split] = {
            "input_ids": sid_seq_dataset.input_ids,
            "attention_mask": sid_seq_dataset.attention_mask,
            "fut_ids": sid_seq_dataset.fut_ids,
        }
    return out


def build_or_load_datasets(
    split_datasets: dict,
    item2sid: Dict[str, List[int]],
    args,
    is_distributed: bool,
    rank: int,
) -> Dict[str, SemanticIDDataset]:
    cache_dir = os.path.join(args.output_path, "_dataset_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = _dataset_cache_key(args, split_datasets, item2sid)
    cache_path = os.path.join(cache_dir, f"semid_{args.category}_{key}.pt")
    rebuild = bool(getattr(args, "rebuild_dataset_cache", False))

    payload=None
    if _is_main(rank):
        if (not rebuild) and os.path.exists(cache_path):
            log(f"[DATASET] Cache hit, loading tokenised splits from {cache_path}")
            payload = torch.load(cache_path, map_location="cpu")
        else:
            if rebuild and os.path.exists(cache_path):
                log(f"[DATASET] --rebuild_dataset_cache set; ignoring {cache_path}")
            log(f"[DATASET] Building tokenised splits (cache key={key})...")
            payload = _build_split_tensors(split_datasets, item2sid, args.num_codebooks, args.max_seq_length)
            tmp_path = cache_path + ".tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, cache_path)
            log(f"[DATASET] Cached tokenised splits to {cache_path}")

    if is_distributed:
        dist.barrier()
        if not _is_main(rank):
            payload = torch.load(cache_path, map_location="cpu")

    return {split: SemanticIDDataset(from_tensors=payload[split]) for split in ("train", "val", "test")}


def create_dataloaders(
    split_datasets: dict,
    item2sid: Dict[str, List[int]],
    args,
    is_distributed: bool,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    datasets = build_or_load_datasets(
        split_datasets=split_datasets,
        item2sid=item2sid,
        args=args,
        is_distributed=is_distributed,
        rank=rank,
    )

    common_kwargs = {
        "num_workers": args.dataloader_num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.dataloader_num_workers > 0,
    }
    if args.dataloader_num_workers > 0:
        common_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor

    if is_distributed:
        train_sampler = DistributedSampler(datasets["train"], num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(datasets["val"], num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(datasets["test"], num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

        train_loader = DataLoader(datasets["train"], batch_size=args.per_device_batch_size, sampler=train_sampler, drop_last=False, **common_kwargs)
        val_loader = DataLoader(datasets["val"], batch_size=args.eval_batch_size, sampler=val_sampler, drop_last=False, **common_kwargs)
        test_loader = DataLoader(datasets["test"], batch_size=args.eval_batch_size, sampler=test_sampler, drop_last=False, **common_kwargs)
    else:
        train_loader = DataLoader(datasets["train"], shuffle=True, drop_last=False, **common_kwargs)
        val_loader = DataLoader(datasets["val"], shuffle=False, drop_last=False, **common_kwargs)
        test_loader = DataLoader(datasets["test"], shuffle=False, drop_last=False, **common_kwargs)
    return train_loader, val_loader, test_loader


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
    item2sid = collision_mitigation(item2sid, args.codebook_size)
    num_codebooks = len(next(iter(item2sid.values())))

    codebooks = torch.tensor(list(item2sid.values()), dtype=torch.long)
    should_check_prefix = args.should_check_prefix or codebooks is not None

    if _is_main(rank):
        log("[DATALOADER] Create Train/Val/Test dataloaders...")
    dataloaders = create_dataloaders(split_datasets, item2sid, args, is_distributed, rank, world_size)
    trainer_config = build_trainer_config(args, dataloaders[0], is_distributed, local_rank)

    total_vocab_size = num_codebooks * args.codebook_size + 1
    cfg = Config.load_config_from_file(args.config_path) # Load encoder/decoder structural hyperparameters from YAML configs
    encoder_cfg, decoder_cfg = cfg.encoder, cfg.decoder
    encoder_cfg["vocab_size"] = decoder_cfg["vocab_size"] = total_vocab_size

    encoder = init_encoder(encoder_cfg.backbone, encoder_cfg)
    decoder = init_decoder(decoder_cfg.backbone, decoder_cfg)
    model = SemanticIDEncoderDecoder(
        num_hierarchies=num_codebooks,
        num_embeddings_per_hierarchy=args.codebook_size,
        encoder=encoder,
        decoder=decoder,
        codebooks=codebooks,
        d_model=encoder_cfg.d_model,
        num_beams=max(args.num_beams, max(trainer_config.eval_topks)),
        loss_func=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=-100),
        check_prefix=should_check_prefix,
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
    parser.add_argument("--num_codebooks", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--eval_topks", type=str, default="5,10")
    parser.add_argument("--should_check_prefix", action="store_true")
    parser.add_argument("--enable_torch_compile", action="store_true")

    args = parser.parse_args()

    gr_pipeline(args)