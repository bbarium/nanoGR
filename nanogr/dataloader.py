from typing import Tuple

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset


def SeqDataset():
    pass


# TODO: 为了将超出max_item_seq_len的序列重建为多条，需要对每条序列进行tokenization
# 目前代码的效率很低，这里可不可以改为batch tokenization的方式。
# TODO: 这里还有pad_len和max_item_seq_len的冲突，这里我为了保留item的SID不被截断使用的是max_item_seq_len
def tokenize_first_n_items(
    item_seq: list, 
    tokenizer: AutoTokenizer,
    num_codebooks: int, 
    max_item_seq_len:int,
    ignored_label: int=-100,
) -> tuple:
    """
    Tokenizes the first n items in the given item_seq.
    The losses for the first n items can be computed by only forwarding once.

    Args:
        item_seq (list): The item sequence that contains the first n items.

    Returns:
        tuple: A tuple containing the tokenized input_ids, attention_mask, labels, and seq_lens.
    """
    
    # input_ids = tokenizer.encode(''.join(item_seq), return_tensors='pt').squeeze()
    history, target = item_seq[:-1], item_seq[-1]

    history_ids = tokenizer.encode('<|sid_start|>' + ''.join(history) + '<|sid_end|><|rec|>', add_special_tokens=False)
    target_ids = tokenizer.encode(target + tokenizer.eos_token, add_special_tokens=False)
    input_ids = history_ids + target_ids
    seq_lens = len(input_ids)
    
    labels = [ignored_label] * len(history_ids) + target_ids.copy()
    attention_mask = [1] * len(input_ids)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_tokens

    # 根据序列长度和SID的码本数量计算需要padding的长度
    if len(history) < max_item_seq_len:
        pad_lens = (max_item_seq_len + 1) * num_codebooks + 4 - seq_lens
        input_ids.extend([tokenizer.pad_token_id] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        labels.extend([ignored_label] * pad_lens)
    
    return input_ids, attention_mask, labels, seq_lens


def tokenize_function(
    original_item_seq: dict,
    item2sid: dict,
    tokenizer: AutoTokenizer,
    split: str,
    max_item_seq_len: int,
) -> dict:
    """

    """
    num_codebooks = len(next(iter(item2sid.values())))
    item_seq = [''.join(item2sid[item]) for item in original_item_seq] # Convert item IDs into SIDs

    if split == 'train':
        n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

        # Tokenize the first n items if len(item_seq) <= max_item_seq_len + 1
        # Add 1 as the target item is not included in the input sequence
        input_ids, attention_mask, labels, seq_lens = tokenize_first_n_items(
            item_seq=item_seq[:min(len(item_seq), max_item_seq_len+1)],
            tokenizer=tokenizer,
            num_codebooks=num_codebooks,
            max_item_seq_len=max_item_seq_len,
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
        }


def tokenize(
    datasets: dict, 
    item2sid: dict, 
    tokenizer: AutoTokenizer,
    max_item_seq_len: int,
) -> dict:
    """
    Tokenizes the given datasets.

    Args:
        datasets (dict): A dictionary of datasets to tokenize.

    Returns:
        dict: A dictionary of tokenized datasets.
    """
    tokenized_datasets = {}
    for split in datasets:
        tokenized_item_seq = []
        for value in datasets[split].values():
            tokenized_item_seq.append(tokenize_function(
                example=value,
                item2sid=item2sid,
                tokenizer=tokenizer,
                split=split,
                max_item_seq_len=max_item_seq_len
            ))
        tokenized_datasets[split] = tokenized_item_seq

    return tokenized_datasets


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


if __name__ == '__main__':
    from dataset import AmazonReviews2014

    raw_dataset = AmazonReviews2014(
        '/home/hzzhang/public_data/amazon_data/AmazonReviews2014', 
        'Toys_and_Games',
        'sentence'
    )
    split_datasets = raw_dataset.split('leave_one_out')