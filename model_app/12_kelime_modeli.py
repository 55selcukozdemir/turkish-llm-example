
import torch
import torch.nn as nn
import torch.optim as optim


# Karaketer Tokenleştirici ve positional encoding 

vocab = (
    ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>', ' '] +
    
    list('abcçdefgğhıijklmnoöprsştuüvyz') +
    list('ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ') +
    list('0123456789') +
    
    ['.',',',';',':','?','!','"',"'",'(',')','[',']','{','}','-','–','—','…','/','\\','|'] +
    
    ['\n']
)

['<PAD>',   # padding (sequence equalization)
 '<UNK>',   # bilinmeyen karakter
 '<BOS>',   # beginning of sentence
 '<EOS>',   # end of sentence
 '<MASK>',  # masking (BERT tarzı)
 ' ']       # boşluk (space)

# char → index
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
vocab_size = len(vocab)


def one_hot_encode_vertical(text):
    """
    return: (vocab_size, len(text)) one-hot matrix (dikey)
    """

    result = np.zeros((vocab_size, len(text)), dtype=np.int32)

    for i, ch in enumerate(text):
        if ch in char_to_idx:
            idx = char_to_idx[ch]
        else:
            idx = char_to_idx['<UNK>']

        result[idx, i] = 1  # 🔥 yön değişti

    return result
def one_hot_encode(text):
    """
    return: (len(text), vocab_size) one-hot matrix
    """

    result = np.zeros((len(text), vocab_size), dtype=np.int32)

    for i, ch in enumerate(text):
        if ch in char_to_idx:
            idx = char_to_idx[ch]
        else:
            idx = char_to_idx['<UNK>']

        result[i, idx] = 1  # 🔥 eski yön

    return result



def get_angle(pos, i, d_model):
    """
    Pozisyon ve boyut indeksine göre açı değerini hesaplar
    """
    return pos / (10000 ** ((2 * (i // 2)) / d_model))


def positional_encoding_matrix(seq_len, d_model):
    """
    Tüm positional encoding matrisini oluşturur
    """
    PE = np.ones((seq_len, d_model))


    for pos in range(seq_len):
        for i in range(d_model):
            angle = get_angle(pos, i, d_model)
            if i % 2 == 0:
                PE[pos, i] = np.sin(angle)
            else:
                PE[pos, i] = np.cos(angle)

    return PE


text = "merhaba"
print(len(text))
encoded = one_hot_encode(text)
pe = positional_encoding_matrix(len(text), vocab_size) # d_model = vocab_size
# np.set_printoptions(threshold=np.inf)
# print(pe)

print(pe.shape )
print(encoded.shape)

pe = pe + encoded



# Model 
class VecModel(nn.Module):
    def __init__(self):
        super(VecModel, self).__init__()

        self.linear1 = nn.Linear(100, 200)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(200, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x



# Eğitim
model = VecModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 100 feature'lı input
X = torch.randn(64, 100)   # batch size = 64
y = torch.randint(0, 10, (64,))  # 10 sınıf

epochs = 10

for epoch in range(epochs):
    model.train()

    # forward
    outputs = model(X)

    # loss
    loss = criterion(outputs, y)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")