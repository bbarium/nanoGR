import os
import math
import json
import numpy as np

from nanogr.dataset import AmazonReviewsDataset
from nanogr.utils import log


class AbstractTokenizer:
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        max_item_seq_len: int,
        output_path: str,
        dataset: AmazonReviewsDataset=None,
    ):
        self.output_path = output_path
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.max_item_seq_len = max_item_seq_len
        self.n_codebook_bits = self._get_codebook_bits(codebook_size)
        
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        self.ignored_label = -100

    @property
    def index_factory(self):
        raise NotImplementedError('Index factory not implemented')

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _generate_semantic_id(self):
        raise NotImplementedError('Tokenization not implemented')

    def embd_to_sid(self):
        raise NotImplementedError('')
    
    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """
        Shift Semantic ID that the tokens can be differentiated between the different codebooks.

        Args:
            item2sem_ids (dict): A dictionary mapping items to their corresponding semantic IDs.

        Returns:
            dict: A dictionary mapping items to their corresponding tokens.
        """
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.num_codebooks):    # "+ 1" as 0 is reserved for padding
                tokens[digit] += self.codebook_size * digit + 1
            item2sem_ids[item] = list(map(lambda t: f'<t_{t}>', tokens))
        return item2sem_ids

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        """
        Tokenizes the first n items in the given item_seq.
        The losses for the first n items can be computed by only forwarding once.

        Args:
            item_seq (list): The item sequence that contains the first n items.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, labels, and seq_lens.
        """
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_item_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = [self.item2id[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        """
        Tokenizes the later items in the item sequence.
        Only the last one items are used as the target item.

        Args:
            item_seq (list): The item sequence.

        Returns:
            tuple: A tuple containing the tokenized input IDs, attention mask, labels, and seq_lens.
        """
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]

        pad_lens = self.max_item_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
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
            n_return_examples = max(len(item_seq) - self.max_item_seq_len, 1)

            # Tokenize the first n items if len(item_seq) <= max_item_seq_len + 1
            # Add 1 as the target item is not included in the input sequence
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                item_seq=item_seq[:min(len(item_seq), self.max_item_seq_len+1)]
            )
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]

            # Tokenize the later items if len(item_seq) > max_item_seq_len + 1
            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i:i+self.max_item_seq_len+1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
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
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(self.max_item_seq_len+1):],
                pad_labels=False
            )
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels[-1:]],
                'seq_lens': [seq_lens]
            }

    def tokenize(self, datasets: dict) -> dict:
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
                lambda t: self.tokenize_function(t, split),
                batched=False,
                remove_columns=datasets[split].column_names,
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')
        return tokenized_datasets
    
    def __textualize_function(self, example: dict, item2tokens: dict):
        item_seq = example['item_seq'][0]
        input_ids = ''
        for item in item_seq[:-1]:
            input_ids += ''.join(item2tokens[item])
        label_ids = ''.join(item2tokens[item_seq[-1]])
        return {'input_ids': [input_ids], 'label_ids': [label_ids]}
    
    def textualize(self, datasets: dict, item2tokens: dict) -> dict:
        textualized_datasets = {}
        
        for split in datasets:
            textualized_datasets[split] = datasets[split].map(
                lambda t: self.__textualize_function(t, item2tokens),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            textualized_datasets[split].set_format(type='torch')
        return textualized_datasets