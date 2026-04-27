from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def build_sid_text(item_ids: list[Any], item2sid: dict[Any, list[str]]) -> list[str]:
    return [''.join(item2sid[item]) for item in item_ids]


class ItemSeqDataset(Dataset):
    def __init__(self, split_dataset: dict, item2sid: dict, split: str):
        super().__init__()
        self.split = split
        self.samples = []

        for user, history, target in zip(
            split_dataset['user'],
            split_dataset['item_seq'],
            split_dataset['fut_item'],
        ):
            history_sids = build_sid_text(history, item2sid)
            target_sids = ''.join(item2sid[target])

            self.samples.append(
                {
                    'user': user,
                    'prompt_text': '<|sid_start|>' + ''.join(history_sids) + '<|sid_end|><|rec|>',
                    'label_text': target_sids,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class ItemSeqCollator:
    def __init__(self, tokenizer: AutoTokenizer, ignored_label: int = -100):
        self.tokenizer = tokenizer
        self.ignored_label = ignored_label

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        prompt_texts = [sample['prompt_text'] for sample in batch]
        label_texts = [sample['label_text'] + self.tokenizer.eos_token for sample in batch]

        prompt_tokens = self.tokenizer(
            prompt_texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        label_tokens = self.tokenizer(
            label_texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        input_ids = []
        labels = []
        attention_masks = []
        max_length = 0

        for prompt_ids, label_ids in zip(prompt_tokens['input_ids'], label_tokens['input_ids']):
            merged_ids = prompt_ids + label_ids
            input_ids.append(merged_ids)
            labels.append([self.ignored_label] * len(prompt_ids) + label_ids.copy())
            max_length = max(max_length, len(merged_ids))

        for index, merged_ids in enumerate(input_ids):
            pad_length = max_length - len(merged_ids)
            input_ids[index] = [self.tokenizer.pad_token_id] * pad_length + merged_ids
            labels[index] = [self.ignored_label] * pad_length + labels[index]
            attention_masks.append([0] * pad_length + [1] * len(merged_ids))

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def create_text_dataloader(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    persistent_workers = num_workers > 0
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': persistent_workers,
        'collate_fn': ItemSeqCollator(tokenizer),
    }

    if persistent_workers:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor

    return DataLoader(**dataloader_kwargs)