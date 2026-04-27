from .tokenizer import AbstractTokenizer


class KMeansTokenizer(AbstractTokenizer):
    def __init__(self, num_codebooks, codebook_size, output_path):
        super().__init__(num_codebooks, codebook_size, output_path)
    
    def forward(self):
        pass
