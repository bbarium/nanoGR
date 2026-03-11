from typing import Tuple

from torch.utils.data import DataLoader
from nanogr.utils import log


def tokenize_first_n_items(
    item_seq: list, 
    item2id: dict, 
    max_item_seq_len:int,
    ignored_label: int,
) -> tuple:
    """
    Tokenizes the first n items in the given item_seq.
    The losses for the first n items can be computed by only forwarding once.

    Args:
        item_seq (list): The item sequence that contains the first n items.

    Returns:
        tuple: A tuple containing the tokenized input_ids, attention_mask, labels, and seq_lens.
    """
    input_ids = [item2id[item] for item in item_seq[:-1]]
    seq_lens = len(input_ids)
    attention_mask = [1] * seq_lens

    pad_lens = max_item_seq_len - seq_lens
    input_ids.extend([0] * pad_lens)
    attention_mask.extend([0] * pad_lens)

    labels = [item2id[item] for item in item_seq[1:]]
    labels.extend([ignored_label] * pad_lens)

    return input_ids, attention_mask, labels, seq_lens


def tokenize_later_items(
    item_seq: list, 
    item2id: dict,
    max_item_seq_len: int,
    ignored_label: int,
    pad_labels: bool = True
) -> tuple:
    """
    Tokenizes the later items in the item sequence.
    Only the last one items are used as the target item.

    Args:
        item_seq (list): The item sequence.

    Returns:
        tuple: A tuple containing the tokenized input IDs, attention mask, labels, and seq_lens.
    """
    input_ids = [item2id[item] for item in item_seq[:-1]]
    seq_lens = len(input_ids)
    attention_mask = [1] * seq_lens
    labels = [ignored_label] * seq_lens
    labels[-1] = item2id[item_seq[-1]]

    pad_lens = max_item_seq_len - seq_lens
    input_ids.extend([0] * pad_lens)
    attention_mask.extend([0] * pad_lens)
    if pad_labels:
        labels.extend([ignored_label] * pad_lens)

    return input_ids, attention_mask, labels, seq_lens


def tokenize_function(
    example: dict, 
    split: str,
    max_item_seq_len: int,
) -> dict:
    """
    Tokenizes the input example based on the specified split.

    Args:
        example (dict): The input example containing the item sequence.
        split (str): The split type ('train' or 'val' or 'test').

    Returns:
        dict: A dictionary containing the tokenized input, attention mask, and labels.
    """
    item_seq = example['item_seq'][0]
    if split == 'train':
        n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

        # Tokenize the first n items if len(item_seq) <= max_item_seq_len + 1
        # Add 1 as the target item is not included in the input sequence
        input_ids, attention_mask, labels, seq_lens = tokenize_first_n_items(
            item_seq=item_seq[:min(len(item_seq), max_item_seq_len+1)]
        )
        all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
            [input_ids], [attention_mask], [labels], [seq_lens]

        # Tokenize the later items if len(item_seq) > max_item_seq_len + 1
        for i in range(1, n_return_examples):
            cur_item_seq = item_seq[i:i+max_item_seq_len+1]
            input_ids, attention_mask, labels, seq_lens = tokenize_later_items(cur_item_seq)
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
            all_seq_lens.append(seq_lens)

        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'labels': all_labels,
            'seq_lens': all_seq_lens,
        }
    else:
        input_ids, attention_mask, labels, seq_lens = tokenize_later_items(
            item_seq=item_seq[-(max_item_seq_len+1):],
            pad_labels=False
        )
        return {
            'input_ids': [input_ids],
            'attention_mask': [attention_mask],
            'labels': [labels[-1:]],
            'seq_lens': [seq_lens]
        }


def tokenize(datasets: dict) -> dict:
    """
    Tokenizes the given datasets.

    Args:
        datasets (dict): A dictionary of datasets to tokenize.

    Returns:
        dict: A dictionary of tokenized datasets.
    """
    tokenized_datasets = {}
    for split in datasets:
        tokenized_datasets[split] = datasets[split].map(
            lambda t: tokenize_function(t, split),
            batched=False,
            remove_columns=datasets[split].column_names,
            desc=f'Tokenizing {split} set: '
        )

    for split in datasets:
        tokenized_datasets[split].set_format(type='torch')

    return tokenized_datasets


def textualize_function(example: dict, item2sid: dict):
        item_seq = example['item_seq'][0]
        input_ids = ''
        for item in item_seq[:-1]:
            input_ids += ''.join(item2sid[item])
        label_ids = ''.join(item2sid[item_seq[-1]])
        
        return {'input_ids': [input_ids], 'label_ids': [label_ids]}


def textualize(datasets: dict, item2sid: dict) -> dict:
    textualized_datasets = {}
    
    for split in datasets:
        textualized_datasets[split] = datasets[split].map(
            lambda t: textualize_function(t, item2sid),
            batched=True,
            batch_size=1,
            remove_columns=datasets[split].column_names,
            desc=f'Tokenizing {split} set: '
        )

    for split in datasets:
        textualized_datasets[split].set_format(type='torch')
    return textualized_datasets


# Create Train, Val and Test Dataloaders
def create_dataloader(
    datasets: dict, 
    train_batch_size: int, 
    eval_batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataloader = DataLoader(
        datasets['train'],
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8
    )
    val_dataloader = DataLoader(
        datasets['val'],
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=8
    )
    test_dataloader = DataLoader(
        datasets['test'],
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=8
    )
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    return dataloaders