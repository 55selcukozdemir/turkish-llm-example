import torch
import torch.nn as nn
import re



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

# -----------------------
# 3. Embedding Modeli
# -----------------------
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# -----------------------
# 4. Token → Embedding
# -----------------------
def tokenize_and_embed(text, model, word_to_idx):
    tokens = tokenize(text)
    indices = torch.tensor([word_to_idx[t] for t in tokens])
    embeddings = model(indices)
    return tokens, embeddings

# -----------------------
# 5. Örnek kullanım
# -----------------------
if __name__ == "__main__":

        # 1. Dosyayı oku
    with open("./cevre.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()

    # Tokenize
    tokens = tokenize(text)

    # Vocab oluştur
    word_to_idx, idx_to_word = build_vocab(tokens)

    # Model oluştur
    vocab_size = len(word_to_idx)
    model = EmbeddingModel(vocab_size, embedding_dim=100)

    # Embedding al
    tokens, embeddings = tokenize_and_embed(text, model, word_to_idx)

    # Sonuçlar
    print("Tokenlar:")
    print(tokens)

    print("\nEmbedding boyutu:")
    print(embeddings.shape)

    print("\nİlk token embedding vektörü:")
    print(embeddings[0])