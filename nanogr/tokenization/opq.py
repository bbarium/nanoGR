import os
import math
import json
import numpy as np

from .tokenizer import AbstractTokenizer

from nanogr.dataset import AmazonReviewsDataset
from nanogr.utils import log


class OPQTokenizer(AbstractTokenizer):
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        max_item_seq_len: int,
        output_path: str,
        dataset: AmazonReviewsDataset=None,
        opq_use_gpu: bool=True,
        opq_gpu_id: int=0,
        num_proc: int = 2,
        faiss_omp_num_threads: int=4,
    ):
        super(OPQTokenizer, self).__init__(num_codebooks, codebook_size, max_item_seq_len, output_path, dataset)
        self.num_proc = num_proc
        self.opq_use_gpu = opq_use_gpu
        self.opq_gpu_id = opq_gpu_id
        self.faiss_omp_num_threads = faiss_omp_num_threads

    @property
    def index_factory(self) -> str:
        return f'OPQ{self.num_codebooks},IVF1,PQ{self.num_codebooks}x{self.n_codebook_bits}'

    def _generate_semantic_id(self, sent_embs, sem_ids_path, train_mask):
        """
        Generates semantic IDs using the OPQ algorithm.

        Args:
            sent_embs (numpy.ndarray): Array of sentence embeddings.
            sem_ids_path (str): Path to save the generated semantic IDs.
            train_mask (numpy.ndarray): Boolean mask indicating the training samples.
        """
        import faiss
        if self.opq_use_gpu:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.num_codebooks >= 56
        faiss.omp_set_num_threads(self.faiss_omp_num_threads)
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        log(f'[TOKENIZER] Training index...')
        if self.opq_use_gpu:
            index = faiss.index_cpu_to_gpu(res, self.opq_gpu_id, index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.opq_use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(self.num_codebooks):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)

        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(pq_codes[i].tolist())
        
        return item2sem_ids
    
    def emb_to_sid(self, sent_embs, training_item_mask) -> dict:
        """
        Initialize the tokenizer.

        Args:
            dataset (AmazonReviewsDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        sem_ids_path = os.path.join(self.output_path, f'sentence-t5-base_{self.index_factory}.sem_ids')

        if not os.path.exists(sem_ids_path):
            item2sem_ids = self._generate_semantic_id(sent_embs, sem_ids_path, training_item_mask)
            log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
            with open(sem_ids_path, 'w') as f:
                json.dump(item2sem_ids, f)
        else:
            log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
            item2sem_ids = json.load(open(sem_ids_path, 'r'))

        item2tokens = self._sem_ids_to_tokens(item2sem_ids)
        return item2tokens
