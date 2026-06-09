import torch
import torch.nn as nn
import math

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
    def __init__(self, hidden_size, max_seq_len=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        # self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0) # KALDIRILDI
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, word_embeddings, token_type_ids=None):
        seq_length = input_ids.size(1)
        
        # Pozisyon ID'leri oluştur (0, 1, 2, ..., seq_length - 1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        # Dinamik word_embeddings kullanarak token embedding'leri seç
        token_embeds = word_embeddings[input_ids] 
        pos_embeds = self.position_embeddings(position_ids)
        type_embeds = self.token_type_embeddings(token_type_ids)
        
        # Tüm embeddingleri topla
        embeddings = token_embeds + pos_embeds + type_embeds
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BERT(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_heads=12, ff_hidden_size=3072, max_seq_len=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.embedding = BERTEmbedding(hidden_size, max_seq_len, type_vocab_size, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, word_embeddings, token_type_ids=None, attention_mask=None):
        # 1. Girdileri embedding katmanından geçir
        x = self.embedding(input_ids, word_embeddings, token_type_ids)
        
        # 2. Üst üste dizilmiş Transformer bloklarından (Encoder) geçir
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
            
        return x

class BertForMaskedLM(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        # Sınıflandırıcı kaldırıldı, bunun yerine word_embeddings ile matris çarpımı yapılacak
        
    def forward(self, input_ids, word_embeddings, token_type_ids=None, attention_mask=None):
        # BERT çıktısını al: (batch_size, seq_len, hidden_size)
        sequence_output = self.bert(input_ids, word_embeddings, token_type_ids, attention_mask)
        
        # Sınıflandırma logitslerini hesaplamak için sequence_output ile word_embeddings çarpılır.
        # word_embeddings: (vocab_size, hidden_size) -> .transpose(0, 1) -> (hidden_size, vocab_size)
        # prediction_scores: (batch_size, seq_len, vocab_size)
        prediction_scores = torch.matmul(sequence_output, word_embeddings.transpose(0, 1))
        
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
