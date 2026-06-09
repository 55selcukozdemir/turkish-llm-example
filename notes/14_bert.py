import torch
import torch.nn as nn
import math
from benimkutuphanem.src.benimkutuphanem import TensorMonitor

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size num_heads'e tam bölünmeli"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Lineer dönüşümler ve head'lere ayırma (Batch, Heads, Seq_Len, Head_Dim)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention: Q * K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # (Batch, Seq_Len) maskesini (Batch, 1, 1, Seq_Len) boyutuna getir
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Attention_Weights * V
        context = torch.matmul(attention_weights, v)
        
        # Head'leri birleştir (Batch, Seq_Len, Hidden_Size)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        output = self.out(context)
        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ff_hidden_size)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ff_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        self.ff = FeedForward(hidden_size, ff_hidden_size, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-Head Attention ve Add & Norm (Residual bağlantı)
        attention_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_out))
        
        # Feed Forward ve Add & Norm (Residual bağlantı)
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        
        # Pozisyon ID'leri oluştur (0, 1, 2, ..., seq_length - 1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        token_embeds = self.token_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        type_embeds = self.token_type_embeddings(token_type_ids)
        
        # Tüm embeddingleri topla
        embeddings = token_embeds + pos_embeds + type_embeds
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, num_heads=12, ff_hidden_size=3072, max_seq_len=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, hidden_size, max_seq_len, type_vocab_size, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # 1. Girdileri embedding katmanından geçir
        x = self.embedding(input_ids, token_type_ids)
        
        # 2. Üst üste dizilmiş Transformer bloklarından (Encoder) geçir
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
            
        return x

class BertForMaskedLM(nn.Module):
    def __init__(self, bert_model, vocab_size, hidden_size):
        super().__init__()
        self.bert = bert_model
        # MLM Classifier: Gizli durumları (hidden_size) kelime dağarcığına (vocab_size) eşler
        self.classifier = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # BERT çıktısını al
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)
        # Tahminler için sınıflandırıcıdan geçir
        prediction_scores = self.classifier(sequence_output)
        return prediction_scores

if __name__ == "__main__":
    import torch.optim as optim
    
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
    for epoch in range(1, 21):
        optimizer.zero_grad()
        
        # İleri besleme (Forward Pass)
        # Çıktı boyutu: (Batch Size, Sequence Length, Vocab Size)
        outputs = model(input_ids, token_type_ids, attention_mask)
        
        # Loss hesaplamak için tensor boyutlarını (Batch * Seq_Len, Vocab_Size) şeklinde düzleştir
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        
        # Geri Yayılım (Backward Pass)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch:02d} | Loss: {loss.item():.4f}")
        
    print("\nEğitim tamamlandı! Loss değeri düştüğü için model maskelenmiş kelimeleri (ezberleyerek de olsa) öğrenmeye başladı.")
