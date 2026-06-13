from vocabulary import VocabularyManager
from tokenizer import Tokenizer
from vocab_model import VocabModel
from data import DataPreparer
from bert_model import BERT, BertForMaskedLM, BertWordEmbedding

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch


from pathlib import Path
from tqdm import tqdm

import numpy as np

# Hiperparametreler
hidden_size = 256       # Hidden state boyutu
num_layers = 2          # Transformer katmanı sayısı
num_heads = 4           # Attention head sayısı
ff_hidden_size = 512    # Feed-Forward katmanının gizli boyutu
max_seq_len = 32        # BERT dizi uzunluğu
char_max_len = 70       # Bir kelimenin maksimum karakter uzunluğu
bit_depth = 8           # Her karakterin kaç bit ile temsil edliceğeini gösterir
feature_length = char_max_len * bit_depth


BASE_DIR = Path(__file__).resolve().parent
test_text_path = BASE_DIR / "dataset" / "cikti.txt"

print("Dosyalar okunuyor ve Tokenizer, Vocabulary hazırlanıyor...")
datapreparer = DataPreparer(test_text_path)




print("Modeller başlatılıyor...")
vocab_model = VocabModel(feature_length, hidden_size)
tokenizer = Tokenizer(test_text_path, hidden_size, vocab_model)

base_bert = BERT(
    vocab_model=vocab_model,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_heads=num_heads,
    ff_hidden_size=ff_hidden_size,
    max_seq_len=max_seq_len,
    feature_length=feature_length
)

model = BertForMaskedLM(base_bert)

# Optimizasyon için hem BERT hem de VocabModel parametreleri verilir
optimizer = optim.Adam(list(model.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()
epochs = 200

for i in tqdm(datapreparer.get_list_of_array()):
    optimizer.zero_grad()
    

    print([VocabularyManager(char_max_len).get_vocab_array(a).shape for  a in Tokenizer.tokenize(i)])
    word_representation = [VocabularyManager(char_max_len).get_vocab_array(a) for a in Tokenizer.tokenize(i)] 
    prediction_scores: torch.Tensor = model(word_representation)
    
    loss = criterion(prediction_scores.view(-1, vocab_size), targets.view(-1))


    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {loss.item():.4f}")
    print(i)




"""

for epoch in range(epochs):
    optimizer.zero_grad()
    # 1. VocabModel ile o anki iterasyon için kelime embedding'lerini hesapla
    # vocab_matrices_tensor boyutu: (vocab_size, char_max_len, bit_depth)
    # word_embeddings boyutu: (vocab_size, hidden_size)
    word_embeddings = vocab_model(vocab_matrices_tensor)
    
    # Dummy input (Örnek batch: 4 cümle, max_seq_len=32)
    # Gerçek eğitimde Tokenizer'dan gelen encode edilmiş cümleleri kullanacaksınız.
    input_ids = torch.randint(0, vocab_size, (4, max_seq_len))
    targets = input_ids.clone() 
    
    # 2. Modeli çalıştır
    # prediction_scores: (batch_size, seq_len, vocab_size)
    prediction_scores: torch.Tensor = model(input_ids, word_embeddings)
    
    # tensor_to_blender_ply(prediction_scores)
    # export_csv(prediction_scores)


    # Loss hesabı: (batch_size * seq_len, vocab_size) vs (batch_size * seq_len)
    loss = criterion(prediction_scores.view(-1, vocab_size), targets.view(-1))
    
    # 3. Geri yayılım (Backpropagation)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {loss.item():.4f}")

print("Eğitim tamamlandı!")
print(f"VocabModel gradyan normu: {vocab_model.linear1.weight.grad.norm().item():.4f}")

# ----------------- MODEL KAYIT KISMI -----------------
print("\n--- Modeller Kaydediliyor ---")
torch.save(model.state_dict(), "bert_model.pth")
torch.save(vocab_model.state_dict(), "vocab_model.pth")
print("Modeller 'bert_model.pth' ve 'vocab_model.pth' olarak kaydedildi.")


"""