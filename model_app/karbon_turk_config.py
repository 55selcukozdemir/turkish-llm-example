
from vocabulary import VocabularyManager
from benimkutuphanem import Logger
from typing import Literal

import os

class KarbonTurkConfig():
    def __init__(self, BASE_DIR: str):
        self.hidden_size = 256       # Hidden state boyutu
        self.num_layers = 2          # Transformer katmanı sayısı
        self.num_heads = 4           # Attention head sayısı
        self.ff_hidden_size = 512    # Feed-Forward katmanının gizli boyutu
        self.max_seq_len = 1000        # BERT dizi uzunluğu
        
        self.char_max_len = 70       # Bir kelimenin maksimum karakter uzunluğu
        self.bit_depth = 8           # Her karakterin kaç bit ile temsil edliceğeini gösterir
        self.feature_length = self.char_max_len * self.bit_depth # Kelimenin olabilecek max uzunluğu.
        
        # Spacial tokens
        self.spacial_tokens_padding = '<PAD>'
        self.spacial_tokens_unknow = '<UNK>'
        
        #Beginning Of Sequence
        self.spacial_tokens_BOS = '<BOS>'
        
        #End Of Sequence
        self.spacial_tokens_EOS = '<EOS>'
        
        self.spacial_tokens_mask = '<MASK>'
        
        
        self.data_set = os.path.join(BASE_DIR, "dataset/cikti.txt") 
        
        
        log_folder = BASE_DIR / "logs"
        self.logger = Logger(log_folder)
        
        self.vacobulary_manager = VocabularyManager(self.char_max_len, self.bit_depth, logger=self.logger)

    
