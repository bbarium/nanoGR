import numpy as np
import os.path as osp

import torch
from torch import device
from transformers import T5EncoderModel, T5Config
from sentence_transformers import SentenceTransformer

from nanogr.dataset import AmazonReviewsDataset
from nanogr.utils import log


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
        sent_embd = np.fromfile(sent_emb_path, dtype=np.float32)
        sent_embd = sent_embd.reshape(-1, sent_emb_dim)
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
    sent_embd.astype(np.float32).tofile(sent_emb_path)
    return sent_embd


def init_encoder(backbone: str, config):
    """Build the encoder backbone used by ``SemanticIDEncoderDecoder``.

    Args:
        backbone: Name of the backbone implementation. Currently only ``'t5'``
            is supported.
        config: A mapping (``dict`` or attribute-style namespace) carrying the
            following fields: ``vocab_size``, ``d_model``, ``d_kv``,
            ``num_heads``, ``dropout``, ``dim_feedforward``, ``num_layers``.
    """
    cfg = config if isinstance(config, dict) else vars(config)
    if backbone == 't5':
        t5_encoder_config = T5Config(
            vocab_size=cfg['vocab_size'],
            d_model=cfg['d_model'],
            d_kv=cfg['d_kv'],
            num_heads=cfg['num_heads'],
            dropout_rate=cfg['dropout'],
            d_ff=cfg['dim_feedforward'],
            num_layers=cfg['num_layers'],
        )
        encoder = T5EncoderModel(t5_encoder_config)
    else:
        raise NotImplementedError()
    return encoder
