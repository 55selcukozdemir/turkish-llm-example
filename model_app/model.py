from model_app.vocabulary import Vocabulary
from model_app.tokenizer import Tokenizer
from model_app.vocab_model import VocabModel
from model_app.data import DataPreparer
from model_app.bert import BERT, BertForMaskedLM
from model_app.tensor_viewer import tensor_to_blender_ply, export_csv
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

print("Eğitim başlıyor...")
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

# ----------------- ISI HARİTASI ÇIKARMA -----------------
print("\n--- VocabModel Isı Haritası (Heatmap) Çıkarılıyor ---")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # linear1 katmanının ağırlıklarını al
    # weight shape: (2048, feature_length)
    weights = vocab_model.linear1.weight.detach().cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(weights[:100, :], cmap='viridis') # Çok büyük olduğu için ilk 100 nöronu çizelim ki anlaşılsın
    plt.title("VocabModel Linear1 Ağırlık Isı Haritası (İlk 100 Nöron)")
    plt.xlabel("Girdi Özellikleri (Karakter x 8-bit)")
    plt.ylabel("Gizli Katman Nöronları (İlk 100)")
    plt.savefig("vocab_model_heatmap.png")
    plt.close()
    print("Isı haritası 'vocab_model_heatmap.png' olarak kaydedildi.")
except ImportError:
    print("Isı haritası oluşturulamadı: 'matplotlib' veya 'seaborn' kütüphaneleri eksik.")
# ----------------- TEST KISMI -----------------
print("\n--- Model Testi ---")
model.eval()
vocab_model.eval()

word_to_idx = {w: idx for idx, w in tokenizer.word_list}
idx_to_word = {idx: w for idx, w in tokenizer.word_list}

import re
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

def test_text(text):
    # Metni kelimelere ayır
    tokens = tokenize(text)
    
    # Kelimeleri indexlere çevir (Sözlükte yoksa en yakınını/rastgele veya es geçebiliriz, burada basitçe atlıyoruz)
    indices = [word_to_idx[t] for t in tokens if t in word_to_idx]
    
    if len(indices) == 0:
        print("Sözlükte bu metne ait kelime bulunamadı!")
        return
        
    # Tensor'a çevir: (batch_size=1, seq_len)
    # Model max_seq_len (32) boyut bekliyor olabilir ancak Transformer doğası gereği seq_len dinamik olabilir.
    # Yine de max_seq_len kadar padding veya kesme yapabiliriz.
    input_tensor = torch.tensor([indices], dtype=torch.long)
    if input_tensor.size(1) > max_seq_len:
        input_tensor = input_tensor[:, :max_seq_len]
        
    with torch.no_grad():
        # Tüm embeddingleri al (test sırasında da değişebilir)
        word_embeddings = vocab_model(vocab_matrices_tensor)
        
        # Modele ver
        predictions = model(input_tensor, word_embeddings)
        
        # En yüksek olasılıklı kelimelerin indexlerini al: (1, seq_len)
        predicted_indices = torch.argmax(predictions, dim=-1)[0].tolist()
        
    # Indexleri metne geri çevir
    predicted_text = " ".join([idx_to_word[idx] for idx in predicted_indices])
    
    print(f"Girdi Metni: '{text}'")
    print(f"Tahmin Edilen Çıktı: '{predicted_text}'")

test_metni = "çevre çok güzel" # Sözlükte olan kelimelerle örnek (cevre.txt dosyasından)
test_text(test_metni)
test_text("Selam")
test_text("Bugün hava nasıl?")
test_text("Ne yapıyorsun?")

