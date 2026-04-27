import os
import gzip
import json
import numpy as np
import os.path as osp
from tqdm import tqdm
from typing import Optional, Tuple
from collections import defaultdict

from .utils import clean_text, log


class AmazonReviewsDataset:
    def __init__(self, root: str, name: str, process_mode: str):
        self.root = root
        self.name = name
        self.process_mode = process_mode
        
        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = None
        self.split_data = None

    def __str__(self) -> str:
        return f'[Dataset] {self.__class__.__name__}\n' \
                f'\tNumber of users: {self.n_users}\n' \
                f'\tNumber of items: {self.n_items}\n' \
                f'\tNumber of interactions: {self.n_interactions}\n' \
                f'\tAverage item sequence length: {self.avg_item_seq_len}'

    @property
    def n_users(self):
        """
        Returns the number of users in the dataset.

        Returns:
            int: The number of users in the dataset.
        """
        return len(self.user2id)

    @property
    def n_items(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.item2id)

    @property
    def n_interactions(self):
        """
        Returns the total number of interactions in the dataset.

        Returns:
            int: The total number of interactions.
        """
        n_inters = 0
        for user in self.all_item_seqs:
            n_inters += len(self.all_item_seqs[user])
        return n_inters

    @property
    def avg_item_seq_len(self):
        """
        Returns the average length of item sequences in the dataset.

        Returns:
            float: The average length of item sequences.
        """
        return self.n_interactions / self.n_users

    @property
    def user2id(self):
        """
        Returns the user-to-id mapping.

        Returns:
            dict: The user-to-id mapping.
        """
        return self.id_mapping['user2id']

    @property
    def item2id(self):
        """
        Returns the item-to-id mapping.

        Returns:
            dict: The item-to-id mapping.
        """
        return self.id_mapping['item2id']
    
    @property
    def max_min_seq_len(self) -> Tuple[int, int]:
        if len(self.all_item_seqs) == 0:
            return (0, 0)
        
        max_length, min_length = 0, np.inf
        for value in self.all_item_seqs.values():
            max_length = max(max_length, len(value))
            min_length = min(min_length, len(value))
        return (max_length, min_length)

    def _download_and_process_raw(self):
        """
        This method should be implemented in the subclass.
        It is responsible for downloading and processing the raw data.
        """
        raise NotImplementedError('This method should be implemented in the subclass')

    def training_item_mask(self):
        if self.split_data is None:
            raise AttributeError("Can't find the attribute ('split_data'), please create the splits first.")
        
        items_for_training = set()
        for item_seq in self.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
    
        log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {self.n_items - 1}')
        mask = np.zeros(self.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[self.item2id[item] - 1] = True
        return mask


class AmazonReviews2014(AmazonReviewsDataset):
    def __init__(
        self, 
        root: str, 
        name: str, 
        process_mode: str=None
    ):
        super(AmazonReviews2014, self).__init__(root, name, process_mode)

        self._check_available_category()
        log(f'[DATASET] Amazon Reviews 2014 for category: {self.name}')

        self.cache_dir = osp.join(self.root, self.name)
        self._download_and_process_raw()

    def _check_available_category(self):
        """
        Checks if the `self.name` is available in the dataset.

        Raises:
            AssertionError: If the specified category is not available.
        """
        available_categories = [
            'Books',
            'Electronics',
            'Movies_and_TV',
            'CDs_and_Vinyl',
            'Clothing_Shoes_and_Jewelry',
            'Home_and_Kitchen',
            'Kindle_Store',
            'Sports_and_Outdoors',
            'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care',
            'Toys_and_Games',
            'Video_Games',
            'Tools_and_Home_Improvement',
            'Beauty',
            'Apps_for_Android',
            'Office_Products',
            'Pet_Supplies',
            'Automotive',
            'Grocery_and_Gourmet_Food',
            'Patio_Lawn_and_Garden',
            'Baby',
            'Digital_Music',
            'Musical_Instruments',
            'Amazon_Instant_Video'
        ]
        assert self.name in available_categories, \
            f'Category "{self.name}" not available. ' \
            f'Available categories: {available_categories}'

    def _parse_gz(self, path: str):
        """
        Parse a gzipped file and yield each line as a dict.

        Args:
            path (str): The path to the gzipped file.

        Yields:
            object: Each line of the gzipped file, parsed as a dict.
        """
        g = gzip.open(path, 'r')
        for l in g:
            l = l.replace(b'true', b'True').replace(b'false', b'False')
            yield eval(l)

    def _load_reviews(self, path: str) -> list:
        """
        Load reviews from a given path.

        Args:
            path (str): The path to the file containing the reviews.

        Returns:
            list: A list of tuples representing the reviews. Each tuple contains the user ID, item ID, and the interaction timestamp.
        """
        log('[DATASET] Loading reviews...')
        reviews = []
        for inter in self._parse_gz(path):
            user = inter['reviewerID']
            item = inter['asin']
            time = inter['unixReviewTime']
            reviews.append((user, item, int(time)))
        return reviews

    def _get_item_seqs(self, reviews: list[tuple]) -> dict:
        """
        Group the reviews by user and sort the items by time.

        Args:
            reviews (list[tuple]): A list of tuples representing the reviews. Each tuple contains the user, item, and time.

        Returns:
            dict: A dictionary where the keys are the users and the values are lists of items sorted by time.
        """
        # Group reviews by user
        item_seqs = defaultdict(list)
        for data in reviews:
            user, item, time = data
            item_seqs[user].append((item, time))

        # Sort items by time
        for user, item_time in item_seqs.items():
            item_time.sort(key=lambda x: x[1])
            item_seqs[user] = [_[0] for _ in item_time]
        return item_seqs

    def _remap_ids(self, item_seqs: dict) -> tuple[dict, dict]:
        """
        Remaps the user and item IDs in the given item sequences dictionary.

        Args:
            item_seqs (dict): A dictionary containing user-item sequences, where the keys are the users and the values are lists of items sorted by time.

        Returns:
            all_item_seqs (dict): A dictionary containing the user-item sequences.
            id_mapping (dict): A dictionary containing the mapping between raw and remapped user and item IDs.
                - user2id (dict): A dictionary mapping raw user IDs to remapped user IDs.
                - item2id (dict): A dictionary mapping raw item IDs to remapped item IDs.
                - id2user (list): A list mapping remapped user IDs to raw user IDs.
                - id2item (list): A list mapping remapped item IDs to raw item IDs.

        Note:
            The remapped user and item IDs start from 1. The ID 0 is reserved for padding `[PAD]`.
        """
        log('[DATASET] Remapping user and item IDs...')
        for user, items in item_seqs.items():
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)
            iids = []           # item id lists
            for item in items:
                if item not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item)
                iids.append(item)
            self.all_item_seqs[user] = iids
        return self.all_item_seqs, self.id_mapping

    def _process_reviews(
        self,
        input_path: str,
        output_path: str
    ) -> tuple[dict, dict]:
        """
        Process the reviews from the input path and save the data to the output path.

        Args:
            input_path (str): The path to the input file containing the reviews.
            output_path (str): The path to save the data.

        Returns:
            all_item_seqs (dict): A dictionary containing the user-item sequences.
            id_mapping (dict): A dictionary containing data maps.
        """
        # Check if the processed data already exists
        seq_file = osp.join(output_path, 'all_item_seqs.json')
        id_mapping_file = osp.join(output_path, 'id_mapping.json')
        if osp.exists(seq_file) and osp.exists(id_mapping_file):
            log('[DATASET] Reviews have been processed...')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping

        log('[DATASET] Processing reviews...')
        reviews = self._load_reviews(input_path)
        item_seqs = self._get_item_seqs(reviews)
        all_item_seqs, id_mapping = self._remap_ids(item_seqs)

        # Save data
        log('[DATASET] Saving mapping data...')
        with open(seq_file, 'w') as f:
            json.dump(all_item_seqs, f)
        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f)
        return all_item_seqs, id_mapping

    def _load_metadata(self, path: str, item2id: dict) -> dict:
        """
        Load metadata from a given path and filter it based on the provided data maps.

        Args:
            path (str): The path to the metadata file.
            item2id (dict): A dictionary mapping item raw tokens (ASIN) to their corresponding IDs.

        Returns:
            dict: A dictionary containing the filtered metadata.
        """
        log('[DATASET] Loading metadata...')
        data = {}
        item_asins = set(item2id.keys())
        for info in tqdm(self._parse_gz(path)):
            if info['asin'] not in item_asins:
                continue
            data[info['asin']] = info
        return data

    def _sent_process(self, raw: str) -> str:
        """
        Process the raw input according to the raw data type and return a processed sentence.

        Args:
            raw (str): The raw input to be processed.

        Returns:
            str: The processed sentence.
        """
        sentence = ""
        if isinstance(raw, float):
            sentence += str(raw)
            sentence += '.'
        elif len(raw) > 0 and isinstance(raw[0], list):
            for v1 in raw:
                for v in v1:
                    sentence += clean_text(v)[:-1]
                    sentence += ', '
            sentence = sentence[:-2]
            sentence += '.'
        elif isinstance(raw, list):
            for v1 in raw:
                sentence += clean_text(v1)
        else:
            sentence = clean_text(raw)
        return sentence + ' '

    def _extract_meta_sentences(self, metadata: dict) -> dict:
        """
        Extracts meta sentences from the given metadata dictionary.

        Args:
            metadata (dict): A dictionary containing metadata information for each item.

        Returns:
            dict: A dictionary mapping items to their corresponding meta sentences.
        """
        log('[DATASET] Extracting meta sentences...')
        item2meta = {}
        for item, meta in tqdm(metadata.items()):
            meta_sentence = ''
            keys = set(meta.keys())
            features_needed = [
                'title', 'price', 'brand',
                'feature', 'categories', 'description'
            ]
            for feature in features_needed:
                if feature in keys:
                    meta_sentence += self._sent_process(meta[feature])
            item2meta[item] = meta_sentence
        return item2meta

    def _process_meta(self, input_path: str, output_path: str) -> Optional[dict]:
        """
        Process metadata based on the specified process type.

        Args:
            input_path (str): The path to the input metadata file.
            output_path (str): The path to save the processed metadata file.

        Returns:
            dict: A dictionary containing the item metadata.

        Raises:
            NotImplementedError: If the metadata processing type is not implemented.
        """
        meta_file = osp.join(output_path, f'metadata.{self.process_mode}.json')
        if osp.exists(meta_file):
            log('[DATASET] Metadata has been processed...')
            with open(meta_file, 'r') as f:
                return json.load(f)

        log(f'[DATASET] Processing metadata, mode: {self.process_mode}')
        
        if self.process_mode is None:
            # No metadata processing required
            return None
        
        item2meta = self._load_metadata(
            path=input_path,
            item2id=self.item2id
        )
        if self.process_mode == 'sentence':
            # Extract sentences from metadata
            item2meta = self._extract_meta_sentences(metadata=item2meta)
        else:
            raise NotImplementedError('Metadata processing type not implemented.')

        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)
        return item2meta

    def _download_and_process_raw(self):
        """
        Downloads and processes the raw data files.

        This method downloads the raw data files for reviews and metadata from the specified path,
        processes the raw data, and saves the processed data in the cache directory.

        Returns:
            None
        """
        # Load AmazonReivew from raw data files
        raw_data_path = osp.join(self.cache_dir, 'raw')
        reviews_localpath = osp.join(raw_data_path, f'reviews_{self.name}_5.json.gz') 
        meta_localpath = osp.join(raw_data_path, f'meta_{self.name}.json.gz')
        if not (osp.exists(reviews_localpath) or osp.exists(meta_localpath)):
            raise FileNotFoundError(f"Can't find {reviews_localpath} & {meta_localpath}")

        # Following https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/data/data_process.py
        np.random.seed(12345)

        # Process raw data
        processed_data_path = osp.join(self.cache_dir, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)

        self.all_item_seqs, self.id_mapping = self._process_reviews(
            input_path=reviews_localpath,
            output_path=processed_data_path
        )

        self.item2meta = self._process_meta(
            input_path=meta_localpath,
            output_path=processed_data_path
        )
    
    def _leave_one_out(self, all_item_seqs: dict, max_seq_len: int) -> dict:
        """
        Splits the dataset into train, validation, and test sets using the leave-one-out strategy.

        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
                  Each dataset is represented as a dictionary with 'user' and 'item_seq' keys.
                  The 'user' key contains a list of users, and the 'item_seq' key contains a list of item sequences.
        """
        datasets = {
            'train': {'user': [], 'item_seq': [], 'fut_item': []},
            'val':   {'user': [], 'item_seq': [], 'fut_item': []},
            'test':  {'user': [], 'item_seq': [], 'fut_item': []}
        }
        for user in all_item_seqs:
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(all_item_seqs[user][-(max_seq_len+1):-1])
            datasets['test']['fut_item'].append(all_item_seqs[user][-1])
            if len(all_item_seqs[user]) > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(all_item_seqs[user][-(max_seq_len+2):-2])
                datasets['val']['fut_item'].append(all_item_seqs[user][-2])
            if len(all_item_seqs[user]) > 2:
                datasets['train']['user'].append(user)
                datasets['train']['item_seq'].append(all_item_seqs[user][-(max_seq_len+3):-3])
                datasets['train']['fut_item'].append(all_item_seqs[user][-3])
        return datasets


    def split(self, split_strategy: str=None, max_seq_len: int=-1) -> dict:
        """
        Split the dataset into train, validation, and test sets based on the specified split strategy.

        Returns:
            datasets (dict): A dictionary containing the train and test datasets.
        """
        if split_strategy in ['leave_one_out']:
            datasets = self._leave_one_out(self.all_item_seqs, max_seq_len)
        else:
            raise NotImplementedError(f'Split strategy [{split_strategy}] not implemented.')

        return datasets