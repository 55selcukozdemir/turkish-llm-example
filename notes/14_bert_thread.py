import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np

from b14_bert import BertForMaskedLM
from b14_bert import BERT

from pyqtgraph.Qt import QtCore
from benimkutuphanem import TensorMonitor
import pyqtgraph as pg

class TrainerThread(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(object)
    log_signal = QtCore.pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = True
        
    def run(self):
        # ---- Örnek Kullanım ve Eğitim Testi ----

        # Model Parametreleri (Hafif bir model için)
        vocab_size = 1000       # Örnek için daha küçük bir kelime dağarcığı
        hidden_size = 128       # Hidden state boyutu
        num_layers = 2          # Transformer katmanı sayısı
        num_heads = 4           # Attention head sayısı
        ff_hidden_size = 512    # Feed-Forward katmanının gizli boyutu
        max_seq_len = 32        # Maksimum dizi uzunluğu

        print("BERT modeli ve MLM Başlığı başlatılıyor...")
        base_bert = BERT(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_hidden_size=ff_hidden_size,
            max_seq_len=max_seq_len
        )

        model = BertForMaskedLM(base_bert, vocab_size, hidden_size)

        # Optimizasyon ve Kayıp Fonksiyonu
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # Maskelenmiş padding (0) id'leri loss'ta göz ardı edilebilir (ignore_index=0 varsayımıyla)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # --- Örnek Veri Oluşturma (Masked Language Modeling) ---
        batch_size = 4
        seq_len = 16

        # Rastgele cümleler (1 ile vocab_size arası token id'leri, 0'ı padding/ignore_index olarak bıraktık)
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

        # Etiketler (Labels): Gerçek token'lar.
        labels = input_ids.clone()

        # Rastgele maskeleme işlemi (örneğin vocab_size-1 token'ı [MASK] token'ı olsun)
        mask_token_id = vocab_size - 1
        # Her cümlede %15 ihtimalle maskeleme yap
        probability_matrix = torch.full(labels.shape, 0.15)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Maskelenen yerleri mask_token_id yap
        input_ids[masked_indices] = mask_token_id

        # Maskelenmeyen yerlerin loss'a dahil edilmemesi için labels içindeki değerlerini 0 yap
        labels[~masked_indices] = 0 

        attention_mask = torch.ones((batch_size, seq_len)) 
        token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

        print(f"Modeldeki toplam parametre sayısı: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        print("\nEğitim Başlıyor (Örnek 20 Epoch)...")
        model.train()
        for epoch in range(1, 50000):
            if not self.running:
                break
            optimizer.zero_grad()
            
            # İleri besleme (Forward Pass)
            # Çıktı boyutu: (Batch Size, Sequence Length, Vocab Size)
            outputs = model(input_ids, token_type_ids, attention_mask)
            weight = dict(model.named_parameters())["bert.embedding.token_type_embeddings.weight"]
            params_info = [(f"{name} - {len(param.shape)} - {param.shape}") for name, param in model.named_parameters()]
            self.data_signal.emit(weight.detach().cpu().numpy())
            # Loss hesaplamak için tensor boyutlarını (Batch * Seq_Len, Vocab_Size) şeklinde düzleştir
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            
            # Geri Yayılım (Backward Pass)
            loss.backward()
            optimizer.step()
            
            self.log_signal.emit(f"Epoch {epoch} | Loss {loss.item():.4f}")
        self.log_signal.emit("Eğitim tamamlandı")
            
        # print("\nEğitim tamamlandı! Loss değeri düştüğü için model maskelenmiş kelimeleri (ezberleyerek de olsa) öğrenmeye başladı.")
    
    def stop(self):
        self.running = False
    
app = pg.mkQApp("Monitor")

monitor = TensorMonitor()

trainer = TrainerThread()

# Signal → GUI update
trainer.data_signal.connect(monitor.guncelle)

# Log
trainer.log_signal.connect(print)

trainer.start()

app.exec_()