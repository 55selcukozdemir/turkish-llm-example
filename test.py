import torch
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model num_heads'e kalansız bölünmeli"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        pass

    def forward(self, x, mask):
        pass


class FeedForward(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

class EncoderLayer2(nn.Module):
    def __init__(self):
        # attention
        # feed forward
        
        pass
    def forward(self, x, mask):
        # attention = 
        pass





class Bert2(nn.Module):
    def __init__(self, vocabulary_length, d_model, num_layers = 12, num_heads = 12, dropout = 0.1):
        self.token_embedding = nn.Embedding(vocabulary_length, d_model)
        self.positional_embedding = nn.Embedding(vocabulary_length, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)

        self.norm = nn.Linear(d_model)
        self.dropout  = nn.Dropout(dropout)
        
        # aşaığda ki gibi birden fazla encoder katmanı oluşturulmasını bekliyorum.
        # ["" for _ in range(num_layers)]
        self.layers = nn.ModuleList([])

    def forward(self, x, segment_info, mask = None):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, dtype=torch.long, device = x.device)
        positions = positions.unsqueeze(0).expand_as(x)

        idx_embeddings = self.token_embedding(x)
        pos_embeddings = self.positional_embedding(positions)
        seg_embeddings = self.segment_embedding(segment_info)

        embeddings = idx_embeddings + pos_embeddings + seg_embeddings
        x = self.dropout(self.norm(embeddings))

        for layer in self.layers:
            x = layer(x, mask)

        return x

