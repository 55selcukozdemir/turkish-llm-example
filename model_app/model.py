from model_app.vocabulary import Vocabulary
from model_app.tokenizer import Tokenizer
from model_app.vocab_model import VocabModel
from model_app.data import DataPreparer
from model_app.bert import BERT, BertForMaskedLM
import re

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

# Hiperparametreler
hidden_size = 256       # Hidden state boyutu
num_layers = 2          # Transformer katmanı sayısı
num_heads = 4           # Attention head sayısı
ff_hidden_size = 512    # Feed-Forward katmanının gizli boyutu
max_seq_len = 32        # BERT dizi uzunluğu
char_max_len = 70       # Bir kelimenin maksimum karakter uzunluğu
bit_depth = 8

test_text_path = "/Users/55selcukozdemir/Desktop/turkish-llm-example/notes/project/cikti.txt"

print("Dosyalar okunuyor ve Tokenizer, Vocabulary hazırlanıyor...")
datapreparer = DataPreparer(test_text_path)
tokenizer = Tokenizer(test_text_path, hidden_size)


print("Modeller başlatılıyor...")
feature_length = char_max_len * bit_depth
vocab_model = VocabModel(feature_length, hidden_size)

base_bert = BERT(
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_heads=num_heads,
    ff_hidden_size=ff_hidden_size,
    max_seq_len=max_seq_len
)

model = BertForMaskedLM(base_bert)

# Optimizasyon için hem BERT hem de VocabModel parametreleri verilir
optimizer = optim.Adam(list(model.parameters()) + list(vocab_model.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()
epochs = 200

for epoch in range(epochs):

    batchs = datapreparer.get_split_batchs(10)
    train_batchs = []
    # 'y' indeksi için bir alt liste yoksa oluştur
    for y in range(len(batchs)):
        train_batchs.append([]) # Yeni bir alt liste (satır) ekle
        for b in range(len(batchs[y])):
            train_batchs[y].append([]) # Satırın içine elemanı ekle
    
    for y, batch in enumerate(batchs):
        # her seferde birden fazla metin gelecek ve her metnin birden fazla kelimesi olduğu için ayrı ayrı değerlendirmeliyim.

        for i2, b in enumerate(batch):
            all_text = re.findall(r"\w+|[^\w\s]", b) # metni parçaladım
            tmp = []

            vocab_matrices = []
            for i, c in enumerate(all_text): # her cümlenin kelimelerini ilk özellik çıkarımı için işliyorum.
                vocab_matrices.append(vocabulary.get_vocab_array(c))
            
            tmp = vocab_model(torch.tensor(np.array(vocab_matrices), dtype=torch.float32)) # her kelimeye positional encoding benzeri bir şey uyguladım devam ettim.
            
        
            train_batchs[y][i2] = tmp

        
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

