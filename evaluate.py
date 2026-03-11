import math
import argparse
import json
from tqdm import tqdm
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from torch.utils.data import DataLoader

from nanogr.utils import log, seed_everything
from nanogr.dataset import AmazonReviews2014
from finetune import collision_mitigation


class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch_data):
        input_text = [data['input_ids'] for data in batch_data]
        label_text = [data['label_ids'] for data in batch_data]
        
        tokenized_input = self.tokenizer(
            input_text, 
            return_tensors='pt', 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True
        )
        
        return (tokenized_input, label_text)


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(prefix_sequence, self.trie_dict)

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(prefix_sequence: List[int], trie_dict: Dict):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(prefix_sequence[1:], trie_dict[prefix_sequence[0]])
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


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        if sentence[-1] == 151643:
            trie_out = list(candidate_trie.trie_dict.keys())
        else:
            trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    # predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]
    # print(predictions)##################
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
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
    item2sid = json.load(open(args.sem_ids_path, 'r'))
    item2sid = collision_mitigation(item2sid, 256)
       
    # Instantiate a tokenization and add new SIDs
    log(f'[TOKENIZER] Loading the tokenization from {args.ckpt_path}...')
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    # tokenizer.padding_side='left'
    
    log(f'[Dataset] Loading the test dataset...')
    raw_dataset = AmazonReviews2014(args.data_path, args.category, args.metadata)
    split_datasets = raw_dataset.split(args.split_strategy)
    
    tokenized_datasets = torch.load(args.tmp_path, weights_only=False)
    test_dataset = tokenized_datasets['test']
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=Collator(tokenizer))
    log(f'[Dataset] Test Dataset Size: {len(test_dataset)}')
    
    all_items = []
    for item_seq in split_datasets['test']['item_seq']:
        all_items += item_seq
    all_items = set(all_items)
    all_items = [''.join(map(str, item2sid[item])) for item in all_items]
    
    candidate_trie = Trie([tokenizer.encode(candidate) for candidate in all_items])
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
    
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, device_map='auto')

    all_prompt_results, metrics_results = [], {}
    total = 0
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            input, target = batch_data

            input = input.to(device)
            output = model.generate(
                inputs=input['input_ids'],
                attention_mask=input['attention_mask'],
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
            print(output[:5], target[:5])
            topk_res = get_topk_results(output,scores, target, args.num_beams,
                                            all_items=all_items if args.filter_items else None)

            batch_metrics_res = get_metrics_results(topk_res)
            
            total += len(target)
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
    parse.add_argument('--batch_size', type=int, default=128)
    
    # Parameters of local file paths
    parse.add_argument('--data_path', type=str, default='/home/hzzhang/public_data/amazon_data/AmazonReviews2014')
    parse.add_argument('--sem_ids_path', type=str, default='/home/hzzhang/nanoGR/outputs/sentence-t5-base_Toys_and_Games_RQVAE3x256.dim32.sem_ids')
    parse.add_argument('--ckpt_path', type=str, default='/home/hzzhang/nanoGR/backup/checkpoint-7742')
    parse.add_argument('--gr_path', type=str, default='/home/hzzhang/Checkpoints/Qwen3-0.6B')
    parse.add_argument('--tmp_path', type=str, default='./tokenized_datasets.pt')
    
    # Parameters of the dataset
    parse.add_argument('--category', type=str, default='Toys_and_Games')
    parse.add_argument('--metadata', type=str, default='sentence')
    parse.add_argument('--split_strategy', type=str, default='leave_one_out')
    
    parse.add_argument('--num_beams', type=int, default=10)
    parse.add_argument('--filter_items', type=bool, default=True)
    args = parse.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # while True:
    evaluate()