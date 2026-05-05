
from numpy import dtype

import torch.nn as nn
import torch

import numpy as np
import math

import matplotlib

#5 FeedForward 

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

#4 
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden size num_heads'e tam bölünmeli"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: nn.Linear, k, v, mask = None):
            batch_size = q.size(0)
            q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt (self.head_dim)

            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask == 0, -1e9)

            attention_weights = torch.softmax(scores, dim =-1)

            attention_weights = self.dropout(attention_weights)

            context = torch.matmul(attention_weights, v)

            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

            output = self.out(context)
            return output

#3 Transformers Block 

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_hidden_size, dropout=0.1):
        super().__init__()
        # self attention 
        # norm
        # feedforward
        # norm

        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)

        self.norm1 = nn.LayerNorm(hidden_size)

        self.ff = FeedForward(hidden_size, ff_hidden_size, dropout)

        self.norm2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attention_out = self.attention(x,x,x,mask)
        x = self.norm1(x + self.dropout(attention_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


#2 Embedding modülünü inşaa edicem.

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.positional_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids = None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        token_embedings = self.token_embeddings(input_ids)
        pos_embeddings = self.positional_embeddings(position_ids)
        type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = token_embedings + pos_embeddings + type_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

#1  Yukarıdan aşağıya doğru inşaa edicem. 
class Bert(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, num_heads=12, ff_hidden_size=3072, max_seq_len=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbedding(vocab_size, hidden_size, max_seq_len, type_vocab_size, dropout) # Kelime vektörlerini tutacağız.
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_hidden_size, dropout) 
            for _ in range(num_layers)
        ]) # Transformer bloklarını tutuyoruz. 



    def forward(self, input_ids, token_type_ids=None, attention_mask = None): # This expect to return a value.
        # Embedding katmanına girdileri veriyorum. Bana her kelimenin embedinglerini 
        # vermesi için.
        x = self.embeddings(input_ids, token_type_ids)
        
        for layer in self.layers:
            # her layer'a embeddingleri veriyorum. aslında
            # her katman bir döngü halinde bir önceki katmanın verisini alıyor.
            x = layer(x)
        return x


class BertForMaskedLMa(nn.Module):
    def __init__(self, bert_model, vocab_size, hidden_size):
        super().__init__()
        self.bert = bert_model
        self.classifer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)

        prediction_scores = self.classifer(sequence_output)

        return prediction_scores


if __name__ == "__main__":
    import torch.optim as optim 

    # örnek kullanım eğitim seti


    vocab_size = 1000
    hidden_size = 128

    num_layers = 2
    num_heads = 4

    ff_hidden_size = 512
    max_seq_len = 32

    print("Bert modeli ve mlm başlığı başlatılıyor")

    base_bert = Bert(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden_size=ff_hidden_size,
        max_seq_len=max_seq_len
    )

    model = BertForMaskedLMa(base_bert, vocab_size, hidden_size)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    batch_size = 4
    seq_len = 16

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

    labels = input_ids.clone()

    mask_token_id = vocab_size - 1

    probability_matrix = torch.full(labels.shape, 0.15)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    input_ids[masked_indices] = mask_token_id

    labels[~masked_indices] = 0


    attention_mask = torch.ones((batch_size, seq_len))
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    print(f"Modeldeki toplam parametre sayısı: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    model.train()

    for epoch in range(1, 21):

        optimizer.zero_grad()

        outputs = model(input_ids, token_type_ids, attention_mask)

        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

        loss.backward()

        optimizer.step()
    
        print(f"Epoch: {epoch:02d} | Loss: {loss.item():.4f}")


    print("\nEğitim tamamlandı! Loss değeri düştüğü için model maskelenmiş kelimeleri (ezberleyerek de olsa) öğrenmeye başladı.")

