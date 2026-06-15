import numpy as np
import torch
import torch.nn as nn
import re
import json

from vocab_model import VocabModel

class Tokenizer():
    def __init__(self, text_file_path, d_model, vocab_model: VocabModel):
        self.vocab_model = vocab_model
        self.text_file_path = text_file_path
        # Dosya yolunu parametre olarak kullanalım
        with open(self.text_file_path, "r", encoding="utf-8") as f:
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
    