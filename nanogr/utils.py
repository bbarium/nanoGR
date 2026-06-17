import argparse
import html
import os
import random
import re
import yaml
import os.path as osp
import numpy as np
from collections import defaultdict
from typing import Dict, List
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Union
from sklearn.cluster import KMeans

import logzero
from logzero import logger

import torch
from torch.types import Tensor


NEW_TOKENS = ['<|sid_start|>', '<|sid_end|>', '<|rec|>']


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_log_file(log_dir: str, filename: str=None):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    if filename is None:
        filename = datetime.strftime(datetime.now(), "%Y%m%d") + ".log"
    logzero.logfile(os.path.join(log_dir, filename))


def log(message, level='info'):
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)
        else:
            raise ValueError(f'Invalid log level: {level}')


class Config(dict):
    def __init__(self, init_dict=None):
        init_dict = {} if init_dict is None else init_dict
        init_dict = self._create_config_tree_from_dict(init_dict)
        super(Config, self).__init__(init_dict)
    
    @classmethod
    def load_config_from_file(cls, args_dir: Union[str, Path]):
        assert osp.exists(args_dir), "Error: The configuration path is not found!"
        
        _, file_extention = osp.splitext(args_dir)
        if file_extention in [".yml", ".yaml"]:
            with open(args_dir, "r") as cfg_file:
                cfg_as_dict = yaml.safe_load(cfg_file)
                return cls(cfg_as_dict)
        else:
            raise NotImplementedError
    
    @classmethod
    def _create_config_tree_from_dict(cls, init_dict: dict):
        dic = deepcopy(init_dict)
        
        for (key, value) in dic.items():
            if isinstance(value, dict):
                dic[key] = cls(value)
        return dic
    
    def merge_from_dict(self, init_dict: dict):
        if init_dict is None:
            return -1
        
        def retrieve_items(key, value, target_dict):
            for key_t in target_dict.keys():
                if key_t == key:
                    target_dict[key_t] = value
                    break
                
                if isinstance(target_dict[key_t], dict):
                    retrieve_items(key, value, target_dict[key_t])
    
        dic = deepcopy(init_dict)
        for key, value in dic.items():
            retrieve_items(key, value, self)
        return 1
            
    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)


def list_to_str(input_list: Union[list, str], remove_blank=False) -> str:
    """
    Converts a list or a string to a string representation.

    Args:
        l (Union[list, str]): The input list or string.

    Returns:
        str: The string representation of the input.
    """
    ret = ''
    if isinstance(input_list, list):
        ret = ', '.join(map(str, input_list))
    else:
        ret = input_list
    if remove_blank:
        ret = ret.replace(' ', '')
    return ret


def clean_text(raw_text: str) -> str:
    """
    Cleans the raw text by removing HTML tags, special characters, and extra spaces.

    Args:
        raw_text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text


def kmeans(feat: Tensor, num_clusters: int, num_iters: int=100) -> Tensor:
    device = feat.device
    x = feat.cpu().detach().numpy()

    cluster = KMeans(n_clusters=num_clusters, n_init=10, max_iter=num_iters).fit(x)
    centers = torch.from_numpy(cluster.cluster_centers_).to(device)
    return centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(- distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q


def collision_mitigation(item2sid: Dict[str, List[int]], codebook_size: int = 256) -> Dict[str, List[int]]:
    new_item2sid = deepcopy(item2sid)

    sid_counts = defaultdict(int)
    for key, value in new_item2sid.items():
        prefix = "_".join(map(str, value))
        new_item2sid[key] = value + [sid_counts[prefix]]
        sid_counts[prefix] += 1

    max_collision = max(sid_counts.values()) if sid_counts else 0
    if max_collision > codebook_size:
        raise ValueError(f"The maximum number of conflicting SIDs ({max_collision}) is greater than the pre-defined codebook size ({codebook_size}).")
    
    for item in new_item2sid.keys():
        sids = []
        for digit in range(len(new_item2sid[item])):
            sid = new_item2sid[item][digit] + codebook_size * digit + 1
            sids.append(sid)
        new_item2sid[item] = sids
    return new_item2sid
