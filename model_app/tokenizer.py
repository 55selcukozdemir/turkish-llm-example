import numpy as np
import torch
import torch.nn as nn
import re
import json




from benimkutuphanem import Logger
from vocabulary import VocabularyManager
from vocab_model import VocabModel
from karbon_turk_config import KarbonTurkConfig

class Tokenizer():
    
    def __init__(self, config:KarbonTurkConfig , vocab_model: VocabModel):
        self.logger : Logger = config.logger
        self.config = config
        self.vocab_model = vocab_model
        
        # Dosya yolunu parametre olarak kullanalım
        with open(self.config.data_set, "r", encoding="utf-8") as f:
            text = f.read().lower()
        
        all_text = self.tokenize(text)

        unic_vocab = sorted(set(all_text)) # tokens yerine all_text

        self.word_list = [ w for w in enumerate(unic_vocab)]

    def save_snapshot(self, path):
        # JSON'a kaydetmek için sözlük (dictionary) yapısı oluşturalım
        snapshot_data = []
        
        for idx, word in self.word_list:
            snapshot_data.append({
                "word": word,
                "index": idx,
                "embedding": self.vocab_model(word)
            })
            
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, ensure_ascii=False, indent=4)
        print(f"Snapshot başarıyla '{path}' konumuna kaydedildi.")

    def get_all_text_embeddings(self):
        snapshot_data = []
        
        for idx, word in self.word_list:
            snapshot_data.append({
                "word": word,
                "index": idx,
                "embedding": self.vocab_model(word)
            })
        return snapshot_data
        
    def get_word(self, index):
        # word_list tuple'lardan (index, word) oluştuğu için 1. elemanı (word) döndürüyoruz
        if 0 <= index < len(self.word_list):
            return self.word_list[index][1]
        return None
    
    @staticmethod
    def tokenize(text):
        # kelime + noktalama koru
        return re.findall(r"\w+|[^\w\s]", text)
    
    
    def get_line_word_representation(self, line_str_list: list[str]):
    
        all_line_presentaiton: np.ndarray = []        
        for x, line_str in enumerate(line_str_list):
            tokens = Tokenizer.tokenize(line_str)
            tmp_line_presentation: np.ndarray = []
            for y in tokens:
                token_presentation = self.config.vacobulary_manager.get_vocab_array(y)
                tmp_line_presentation.append(token_presentation.flatten()) 
                # self.logger.log_ndarray(F"token_{y}", token_presentation)


            all_line_presentaiton.append(tmp_line_presentation)
        return np.array(all_line_presentaiton)
        
        
        
    