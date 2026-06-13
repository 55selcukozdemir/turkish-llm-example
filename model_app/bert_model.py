import torch
import torch.nn as nn
import math

from vocab_model import VocabModel
from vocabulary import VocabularyManager

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

# still missing linear layer
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

# class BERTEmbedding(nn.Module):
#     def __init__(self, hidden_size, embedding_model: nn.Module, max_seq_len=512, type_vocab_size=2, dropout=0.1):
#         super().__init__()
#         # self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0) # KALDIRILDI
#         self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
#         self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
#         self.norm = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, input_ids, word_embeddings, token_type_ids=None):
#         seq_length = input_ids.size(1)
        
#         # Pozisyon ID'leri oluştur (0, 1, 2, ..., seq_length - 1)
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)
            
#         # Dinamik word_embeddings kullanarak token embedding'leri seç
#         token_embeds = word_embeddings[input_ids] 
#         pos_embeds = self.position_embeddings(position_ids)
#         type_embeds = self.token_type_embeddings(token_type_ids)
        
#         # Tüm embeddingleri topla
#         embeddings = token_embeds + pos_embeds + type_embeds
#         embeddings = self.norm(embeddings)
#         embeddings = self.dropout(embeddings)
        
#         return embeddings
    
    
class BertWordEmbedding(nn.Module):
    def __init__(self,  feature_length: int, hidden_size: int, vocab_model: VocabModel):
        """
        Args:
            feature_length (int): presentation size of word
            hidden_size (int): embedding size
        """
        super().__init__()
        self.vocab_model = vocab_model
        self.vocabulary = VocabularyManager(70, 8)
        
    def forward(self, word_representings):
        """
        Args:
            words: word list 
        """
        outputs = [self.vocab_model(a) for a in word_representings]
        outputs = torch.stack(outputs)
        return outputs


class BERT(nn.Module):
    def __init__(self, vocab_model: VocabModel, hidden_size=768, num_layers=12, num_heads=12, ff_hidden_size=3072, max_seq_len=512, type_vocab_size=2, dropout=0.1, feature_length = 70 * 8):
        super().__init__()
        # self.embedding = BERTEmbedding(hidden_size, max_seq_len, type_vocab_size, dropout)
        
        self.positional_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout)
        self.word_embedding_model = BertWordEmbedding(feature_length, hidden_size, vocab_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, word_representation, token_type_ids=None, attention_mask=None):
        
        word_embeddings = self.word_embedding_model(word_representation) # burada embedding oluşuyor. 
        # 1. Girdileri embedding katmanından geçir
        x = self.positional_encoding(word_embeddings)
        
        # 2. Üst üste dizilmiş Transformer bloklarından (Encoder) geçir
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
            
        return x

class BertForMaskedLM(nn.Module):
    def __init__(self, bert_model: BERT, feature_length = 70 * 8):
        super().__init__()
        self.bert = bert_model
        # Sınıflandırıcı kaldırıldı, bunun yerine word_embeddings ile matris çarpımı yapılacak
        
    def forward(self, word_representation, token_type_ids=None, attention_mask=None):
        # BERT çıktısını al: (batch_size, seq_len, hidden_size)
        sequence_output = self.bert(word_representation, token_type_ids, attention_mask)
        
        # Sınıflandırma logitslerini hesaplamak için sequence_output ile word_embeddings çarpılır.
        # word_embeddings: (vocab_size, hidden_size) -> .transpose(0, 1) -> (hidden_size, vocab_size)
        # prediction_scores: (batch_size, seq_len, vocab_size)

        word_embeddings = self.bert.word_embedding_model(word_representation)
        prediction_scores = torch.matmul(sequence_output, word_embeddings.transpose(2, 1))
        
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
