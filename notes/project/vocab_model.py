import torch
import torch.nn as nn
from vocabulary import Vocabulary
class VocabModel(nn.Module):
    def __init__(self, feature_length, hidden_size):
        super().__init__()

        self.linear1 = nn.Linear(feature_length, 2048)
        self.linear2 = nn.Linear(2048, hidden_size)

        self.relu = nn.ReLU()

    def forward(self, vocab):
        vocab = Vocabulary(70)

        vocab_array = vocab.get_vocab_array(vocab)
        # Gelen vocab (batch_size, max_seq_len, bit_depth) şeklinde olabilir, bunu düzleştir (flatten)
        # if len(vocab.shape) > 2 :
        #     vocab = vocab.view(vocab.size(0), -1)

        vocab = vocab_array.view(vocab_array.size(0), -1)

        x = self.linear1(vocab)
        x = self.relu(x)
        x = self.linear2(x)

        return x