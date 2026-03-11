import numpy as np
from sklearn.metrics import ndcg_score
from typing import Tuple

import torch
from torch.types import Tensor


class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break

        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

def calculate_metrics(eval_preds: Tuple[Tensor, Tensor], compute_result: bool):
    preds, labels = eval_preds
    preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
    
    preds = preds.argsort(axis=-1)  # Sort predictions to get top-k recommendations
    labels = labels.astype(np.int32)  # Convert labels to int32

    # Calculate Hit@k
    def hit_at_k(preds, labels, k):
        # Check if any of the top-k predictions are in the true labels
        return np.mean([np.isin(label, pred[:k]) for label, pred in zip(labels, preds)])

    # Calculate NDCG@k
    def ndcg_at_k(preds, labels, k):
        # For each label and pred, compute the NDCG score at k
        ndcg_scores = []
        for label, pred in zip(labels, preds):
            relevance = np.isin(label, pred[:k]).astype(int)
            ndcg_scores.append(ndcg_score([relevance], [np.arange(k)]))  # Assume relevance is binary
        return np.mean(ndcg_scores)

    # Calculate metrics for different values of k
    hits_at_5 = hit_at_k(preds, labels, 5)
    hits_at_10 = hit_at_k(preds, labels, 10)
    hits_at_15 = hit_at_k(preds, labels, 15)

    ndcg_at_5 = ndcg_at_k(preds, labels, 5)
    ndcg_at_10 = ndcg_at_k(preds, labels, 10)
    ndcg_at_15 = ndcg_at_k(preds, labels, 15)

    return {
        'hit@5': hits_at_5,
        'hit@10': hits_at_10,
        'hit@15': hits_at_15,
        'ndcg@5': ndcg_at_5,
        'ndcg@10': ndcg_at_10,
        'ndcg@15': ndcg_at_15
    }