import math
import argparse
import json
from tqdm import tqdm
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanogr.utils import log, seed_everything
from nanogr.dataset import AmazonReviews2014
from nanogr.dataloader import (
    ItemSeqDataset, 
    create_text_dataloader
)
from finetune import collision_mitigation


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def prefix_allowed_tokens_fn(candidate_trie: Trie, eos_token_id: int=None, rec_token_id: int=None):

    def prefix_allowed_tokens(batch_id, sentence):
        sentence_list = sentence.tolist()
        # find the last position of <|rec|> token in the sentence
        # use rindex to handle cases where <|rec|> appears multiple times in the prompt
        try:
            rec_index = len(sentence_list) - 1 - sentence_list[::-1].index(rec_token_id)
        except ValueError:
            rec_index = len(sentence_list) - 1
        # tokens after <|rec|> are the generated prefix
        generated = sentence_list[rec_index + 1:]
        trie_out = candidate_trie.get(generated)
        # if trie_out is empty, the sid is complete — allow eos to end generation
        if not trie_out:
            return [eos_token_id]
        return trie_out

    return prefix_allowed_tokens


def get_topk_results(predictions, scores, targets, k, all_item_sids=None):
    results = []
    B = len(targets)
    # predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]
    # print(predictions)##################
    if all_item_sids is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_item_sids:
                scores[i] = -1000

    # print(scores)
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        # print(pairs)
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results


def get_metrics_results(topk_results):
    
    def ndcg_k(topk_results, k):
        ndcg = 0.0
        for row in topk_results:
            res = row[:k]
            one_ndcg = 0.0
            for i in range(len(res)):
                one_ndcg += res[i] / math.log(i + 2, 2)
            ndcg += one_ndcg
        return ndcg

    def hit_k(topk_results, k):
        hit = 0.0
        for row in topk_results:
            res = row[:k]
            if sum(res) > 0:
                hit += 1
        return hit
    
    res = {}
    for m in ['hit@1', 'hit@5', 'hit@10', 'ndcg@5', 'ndcg@10']:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def evaluate():
    seed_everything(args.seed)
    
    log(f'[TOKENIZER] Loading semantic IDs from {args.sem_ids_path}...')
    with open(args.sem_ids_path, 'r') as file:
        item2sid = json.load(file)
    item2sid = collision_mitigation(item2sid)
    
    # Instantiate a tokenization and add new SIDs
    log(f'[TOKENIZER] Loading the tokenization from {args.ckpt_path}...')
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)

    raw_dataset = AmazonReviews2014(args.data_path, args.category, args.metadata)
    max_raw_len, min_raw_len = raw_dataset.max_min_seq_len
    log(f'[DATASET] Maximum/Minimum lengths of item sequences: {max_raw_len}, {min_raw_len}')

    log(f'[DATASET] Create Train/Val/Test splits via {args.split_strategy}...')
    max_seq_len = -1 if args.max_seq_length >= max_raw_len else args.max_seq_length
    split_datasets = raw_dataset.split(args.split_strategy, max_seq_len)

    log(f'[MODEL] Loading the model checkpoint from {args.ckpt_path}...')
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, device_map='auto')
    
    test_dataset = ItemSeqDataset(split_datasets['test'], item2sid)
    test_loader = create_text_dataloader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )
    log(f'[DATASET] Test Dataset Size: {len(test_dataset)}')
    
    all_item_sids = [''.join(map(str, sid)) for sid in item2sid.values()]
    candidate_trie = Trie([tokenizer.encode(candidate) for candidate in all_item_sids])
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie, tokenizer.eos_token_id)

    ignored_label = -100
    pad_token_id = tokenizer.pad_token_id

    all_prompt_results, metrics_results = [], {}
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            # batch_data is a dict from ItemSeqCollator: {input_ids, attention_mask, labels}
            full_input_ids = batch_data['input_ids'].to(device)
            full_attention_mask = batch_data['attention_mask'].to(device)
            full_labels = batch_data['labels'].to(device)

            # Separate prompt (history + <|rec|>) from target (label) using labels mask.
            # In ItemSeqCollator: label positions are non-ignored; prompt positions are ignored (-100).
            # The collator uses LEFT padding implicitly for pad, but prompt + label are concatenated left-to-right.
            # We need to crop each sample to its prompt portion for generation.
            label_mask = (full_labels != ignored_label)  # True on target tokens
            # Determine per-sample prompt length (tokens before the first target token) and target length
            batch_size = full_input_ids.size(0)
            seq_len = full_input_ids.size(1)

            prompt_lens = []
            target_token_lists = []
            for i in range(batch_size):
                target_positions = label_mask[i].nonzero(as_tuple=False).squeeze(-1)
                if target_positions.numel() == 0:
                    # No target found; treat the whole sequence as prompt
                    prompt_end = seq_len
                    target_tokens = []
                else:
                    prompt_end = target_positions[0].item()
                    target_tokens = full_labels[i, target_positions].tolist()
                prompt_lens.append(prompt_end)
                target_token_lists.append(target_tokens)

            # Build left-padded prompt tensors so that <|rec|> is right-aligned for generation
            max_prompt_len = max(prompt_lens)
            input_ids = torch.full(
                (batch_size, max_prompt_len), pad_token_id,
                dtype=full_input_ids.dtype, device=device,
            )
            attention_mask = torch.zeros(
                (batch_size, max_prompt_len),
                dtype=full_attention_mask.dtype, device=device,
            )
            for i in range(batch_size):
                p_len = prompt_lens[i]
                if p_len == 0:
                    continue
                # Right-align prompt within max_prompt_len (left padding)
                input_ids[i, max_prompt_len - p_len:] = full_input_ids[i, :p_len]
                attention_mask[i, max_prompt_len - p_len:] = full_attention_mask[i, :p_len]

            output = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )
            output_ids, scores = output['sequences'][:, -4:], output['sequences_scores']
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Decode targets from labels
            target_texts = []
            for target_tokens in target_token_lists:
                # remove eos_token if present at the end
                if target_tokens and target_tokens[-1] == tokenizer.eos_token_id:
                    target_tokens = target_tokens[:-1]
                target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
                target_texts.append(target_text.strip().replace(' ', ''))

            log(f'Predictive SIDs: {output[:5]}')
            log(f'Target SIDs: {target_texts[:5]}')
            topk_res = get_topk_results(output, scores, target_texts, args.num_beams,
                                            all_item_sids=all_item_sids if args.filter_items else None)

            batch_metrics_res = get_metrics_results(topk_res)
            
            total += len(target_texts)
            for m, res in batch_metrics_res.items():
                if m not in metrics_results:
                    metrics_results[m] = res
                else:
                    metrics_results[m] += res

            temp={}
            for m in metrics_results:
                temp[m] = metrics_results[m] / total
            print(temp)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # Shared parameters
    parse.add_argument('--gpu', type=int, default=0)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--per_device_batch_size', type=int, default=32)
    parse.add_argument('--max_seq_length', type=int, default=30)
    parse.add_argument('--dataloader_num_workers', type=int, default=4)
    parse.add_argument('--dataloader_prefetch_factor', type=int, default=4)
    
    # Parameters of local file paths
    parse.add_argument('--data_path', type=str, default='/workspace/team_driver/hzzhang/AmazonReviews')
    parse.add_argument('--sem_ids_path', type=str, default='./outputs/sentence-t5-base_Beauty_RQVAE3x256.dim32.sem_ids')
    parse.add_argument('--ckpt_path', type=str, default='./outputs/checkpoint-3584')
    
    # Parameters of the dataset
    parse.add_argument('--category', type=str, default='Beauty')
    parse.add_argument('--metadata', type=str, default='sentence')
    parse.add_argument('--split_strategy', type=str, default='leave_one_out')
    
    parse.add_argument('--num_beams', type=int, default=10)
    parse.add_argument('--filter_items', type=bool, default=True)
    args = parse.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # while True:
    evaluate()