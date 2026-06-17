import torch.nn as nn
import torch

from vocabulary import VocabularyManager
from karbon_turk_config import KarbonTurkConfig
from benimkutuphanem import Logger
class VocabModel(nn.Module):
    def __init__(self, config: KarbonTurkConfig):
        """
        Args:
            feature_length (int): max char lenght * bit_depth
            hidden_size (int): output size of the last layer
        """
        super().__init__()
        self.vocabulary_manager = config.vacobulary_manager

        self.linear1 = nn.Linear(config.feature_length, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.hidden_size)
        self.softmax = nn.Softmax()

    def forward(self, word_representation):

        x = self.linear1(word_representation)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x