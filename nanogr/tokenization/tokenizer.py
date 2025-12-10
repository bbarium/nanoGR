import math


class AbstractTokenizer:
    def __init__(self, num_codebooks: int, codebook_size: int, output_path: str):
        self.output_path = output_path
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.n_codebook_bits = self._get_codebook_bits(codebook_size)

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
    
    def _sid_shift(self, item2sem_ids: dict) -> dict:
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
    
    def _textualize_function(self, example: dict, item2tokens: dict):
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
                lambda t: self._textualize_function(t, item2tokens),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            textualized_datasets[split].set_format(type='torch')
        return textualized_datasets
    
    def evaluate():
        pass