import torch.nn as nn
import torch
from vocabulary import VocabularyManager
class VocabModel(nn.Module):
    def __init__(self, feature_length: int, hidden_size: int, max_seq_len = 70):
        """
        Args:
            feature_length (int): max char lenght * bit_depth
            hidden_size (int): output size of the last layer
        """
        super().__init__()
        self.vocabulary_manager = VocabularyManager(max_seq_len)

        self.linear1 = nn.Linear(feature_length, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, hidden_size)
        self.softmax = nn.Softmax()

    def forward(self, word_representation):
        if not isinstance(word_representation, torch.Tensor):
                word_representation = torch.tensor(word_representation, dtype=torch.float)
        vocab = word_representation.view(1, -1)

        x = self.linear1(vocab)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x