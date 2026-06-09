import numpy as np
import torch
import torch.nn as nn
import re
import json

class Tokenizer():
    def __init__(self, text_file_path, d_model):
        self.text_file_path = text_file_path
        # Dosya yolunu parametre olarak kullanalım
        with open(self.text_file_path, "r", encoding="utf-8") as f:
            text = f.read().lower()
        
        all_text = re.findall(r"\w+|[^\w\s]", text)
        unic_vocab = sorted(set(all_text)) # tokens yerine all_text

        vocab_size = len(unic_vocab) # vocab_size tanımlandı

        self.word_list = [ w for w in enumerate(unic_vocab)]
        self.initial_enbeddings = torch.rand(vocab_size, d_model) * 2 - 1

    def save_snapshot(self, path):
        # JSON'a kaydetmek için sözlük (dictionary) yapısı oluşturalım
        snapshot_data = []
        
        for idx, word in self.word_list:
            snapshot_data.append({
                "word": word,
                "index": idx,
                "embedding": self.initial_enbeddings[idx].tolist()
            })
            
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, ensure_ascii=False, indent=4)
        print(f"Snapshot başarıyla '{path}' konumuna kaydedildi.")

    def update_weight(self, word, wight):
        # word_list içinde kelimenin index'ini bul
        for idx, w in self.word_list:
            if w == word:
                if not isinstance(wight, torch.Tensor):
                    wight = torch.tensor(wight, dtype=self.initial_enbeddings.dtype)
                
                # Yeni ağırlığın boyutu d_model ile eşleşmeli
                if wight.shape[0] != self.initial_enbeddings.shape[1]:
                    raise ValueError(f"Boyut hatası: Verilen ağırlık {wight.shape[0]} boyutunda, beklenen {self.initial_enbeddings.shape[1]}.")
                
                # İlgili index'teki tensorü güncelle
                self.initial_enbeddings[idx] = wight
                return
        
    def get_word(self, index):
        # word_list tuple'lardan (index, word) oluştuğu için 1. elemanı (word) döndürüyoruz
        if 0 <= index < len(self.word_list):
            return self.word_list[index][1]
        return None
    


# -----------------------
# 1. Tokenizer
# -----------------------
def tokenize(text):
    # kelime + noktalama koru
    return re.findall(r"\w+|[^\w\s]", text.lower())

# -----------------------
# 2. Vocabulary oluştur
# -----------------------
def build_vocab(tokens):
    vocab = sorted(set(tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return word_to_idx, idx_to_word

def encode(text, word_to_idx):
    """Metni alır ve kelimelerin sözlükteki indexlerinden oluşan bir liste döner."""
    tokens = tokenize(text)
    return [word_to_idx[t] for t in tokens if t in word_to_idx]

def decode(indices, idx_to_word):
    """Index listesini alır ve tekrar metne çevirir."""
    return " ".join([idx_to_word[i] for i in indices if i in idx_to_word])

# -----------------------
# 5. Örnek kullanım
# -----------------------
if __name__ == "__main__":

        # 1. Dosyayı oku
    with open("/Users/55selcukozdemir/Desktop/turkish-llm-example/notes/12_project/cevre.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()

    # Tokenize
    tokens = tokenize(text)

    # Vocab oluştur
    word_to_idx, idx_to_word = build_vocab(tokens)

    # Model oluştur
    vocab_size = len(word_to_idx)

    random_embedings = torch.rand(vocab_size, 70) * 2 - 1

    # Embedding'leri kelimelerle eşleştirip sözlüğe çevir, index'i de dahil et
    embeddings_dict = {
        word: {
            "index": idx,
            "embedding": random_embedings[idx].tolist()
        } for word, idx in word_to_idx.items()
    }

    # JSON dosyasına yaz
    output_path = "/Users/55selcukozdemir/Desktop/turkish-llm-example/notes/12_project/embeddings.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings_dict, f, ensure_ascii=False, indent=4)
    
    print(f"Embedding değerleri '{output_path}' dosyasına kaydedildi.")