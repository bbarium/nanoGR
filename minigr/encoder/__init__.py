import numpy as np
import os.path as osp

import torch
from torch import device

from sentence_transformers import SentenceTransformer

from minigr.dataset import AmazonReviewsDataset
from minigr.utils import log


def encode_sent_to_emb(
    dataset: AmazonReviewsDataset,
    sent_emb_model: str,
    batch_size: int,
    device: device=None,
    sent_emb_dim: int=768,
    output_path: str='outputs',
):
    """
    Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

    Args:
        dataset (AmazonReviewsDataset): The dataset containing the sentences to encode.
        output_path (str): The path to save the encoded sentence embeddings.

    Returns:
        numpy.ndarray: The encoded sentence embeddings.
    """
    sent_emb_path = osp.join(output_path, f'{dataset.name}.sentence-t5-base.sent_emb')
    if osp.exists(sent_emb_path):
        log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
        sent_embd = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, sent_emb_dim)
    else:
        log(f'[TOKENIZER] Encoding sentence embeddings...')    
        assert dataset.process_mode == 'sentence', 'TIGERTokenizer only supports sentence metadata.'

        meta_sentences = [] # 1-base, meta_sentences[0] -> item_id = 1
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        if 'sentence' in sent_emb_model:
            sent_emb_model = SentenceTransformer(sent_emb_model).to(device)

            sent_embd = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=True,
                device=device
            )
        elif 'cf' in sent_emb_model:
            pass
        else:
            raise NotImplementedError()
    log(f'[TOKENIZER] Sentence embeddings shape: {sent_embd.shape}')
    sent_embd.tofile(sent_emb_path)
    return sent_embd