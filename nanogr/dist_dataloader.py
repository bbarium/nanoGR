import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from nanogr.utils import log


def _is_main(rank: int) -> bool:
    return rank == 0


class SemanticIDDataset(Dataset):
    def __init__(
        self,
        split_dataset: dict=None,
        item2sid: dict=None,
        num_hierarchies: int=None,
        max_history_len: int=None,
    ) -> None:
        assert split_dataset is not None and item2sid is not None
        max_token_len = max_history_len * num_hierarchies

        item_keys = list(item2sid.keys())
        item_to_idx = {k: i for i, k in enumerate(item_keys)}
        sid_table = np.asarray([item2sid[k] for k in item_keys], dtype=np.int64)
        assert sid_table.shape[1] == num_hierarchies, \
            f"item2sid hierarchies ({sid_table.shape[1]}) != num_hierarchies ({num_hierarchies})"

        item_seqs, fut_items, n_samples = split_dataset["item_seq"], split_dataset["fut_item"], len(item_seqs)
        input_ids = np.zeros((n_samples, max_token_len), dtype=np.int64)
        attention_mask = np.zeros((n_samples, max_token_len), dtype=np.int64)
        fut_ids = np.zeros((n_samples, num_hierarchies), dtype=np.int64)
        show_pbar = _is_main(int(os.environ.get("RANK", 0)))
        for i, (history, target) in enumerate(
            tqdm(
                zip(item_seqs, fut_items), 
                total=n_samples, 
                disable=not show_pbar, 
                desc="[DATASET] tokenize"
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


def _build_split_tensors(
    split_datasets: dict,
    item2sid: dict,
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
    item2sid: dict,
    category: str,
    num_codebooks: int,
    max_seq_length: int,
    output_path: str,
    is_distributed: bool,
    rank: int,
) -> Dict[str, SemanticIDDataset]:
    cache_dir = os.path.join(output_path, "_dataset_cache")
    cache_path = os.path.join(cache_dir, f"semid_{category}.pt")

    payload=None
    if _is_main(rank):
        log(f"[DATASET] Building tokenised splits...")
        payload = _build_split_tensors(split_datasets, item2sid, num_codebooks, max_seq_length)
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
    datasets: dict,
    num_workers: int,
    per_device_batch_size: int,
    eval_batch_size: int,
    prefetch_factor: int,
    is_distributed: bool,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = prefetch_factor

    if is_distributed:
        train_sampler = DistributedSampler(datasets["train"], num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(datasets["val"], num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(datasets["test"], num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

        train_loader = DataLoader(datasets["train"], batch_size=per_device_batch_size, sampler=train_sampler, drop_last=False, **common_kwargs)
        val_loader = DataLoader(datasets["val"], batch_size=eval_batch_size, sampler=val_sampler, drop_last=False, **common_kwargs)
        test_loader = DataLoader(datasets["test"], batch_size=eval_batch_size, sampler=test_sampler, drop_last=False, **common_kwargs)
    else:
        train_loader = DataLoader(datasets["train"], batch_size=per_device_batch_size, shuffle=True, drop_last=False, **common_kwargs)
        val_loader = DataLoader(datasets["val"], batch_size=eval_batch_size, shuffle=False, drop_last=False, **common_kwargs)
        test_loader = DataLoader(datasets["test"], batch_size=eval_batch_size, shuffle=False, drop_last=False, **common_kwargs)
    return train_loader, val_loader, test_loader