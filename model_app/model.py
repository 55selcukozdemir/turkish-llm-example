from vocabulary import VocabularyManager
from tokenizer import Tokenizer
from vocab_model import VocabModel
from data import DataPreparer
from bert_model import BERT, BertForMaskedLM, BertWordEmbedding

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch


from pathlib import Path
from tqdm import tqdm

from benimkutuphanem import Logger

from karbon_turk_config import KarbonTurkConfig
import numpy as np

def test():
    BASE_DIR = Path(__file__).resolve().parent
    
    config =  KarbonTurkConfig(BASE_DIR)
    
    vocab_model = VocabModel(config)
    tokenizer = Tokenizer(config, vocab_model)

    base_bert = BERT(
        vocab_model=vocab_model,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_hidden_size=config.ff_hidden_size,
        max_seq_len=config.max_seq_len,
        feature_length=config.feature_length
    )

    model = BertForMaskedLM(base_bert)

    # Optimizasyon için hem BERT hem de VocabModel parametreleri verilir
    optimizer = optim.Adam(list(model.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    # Veri hazırlanması
    datapreparer = DataPreparer(config)

    data_loader = DataLoader(
                datapreparer, 
                batch_size=128,
                shuffle=True,
                num_workers=4
                )
    
    pbar = tqdm(data_loader)
    for data in pbar:
        optimizer.zero_grad()
        
        word_representationi = data["input"]
        prediction_scores: torch.Tensor = model(word_representationi)

        
        # >> prediction_scores.shape
        # >> torch.Size([11, 1, 1])
        probability_matrix = torch.full(prediction_scores.shape, 1.0)
        loss = criterion(prediction_scores, probability_matrix)


        loss.backward()
        optimizer.step()
        # print(i)
        # print(f"- Loss: {loss.item():.4f}")
        # print(f"- Loss: {loss.item()}")
        pbar.set_postfix(loss=loss.item())


    # print("a")
    # gozluk = base_bert.word_embedding_model(VocabularyManager(char_max_len).get_vocab_array("gözlük"))
    # gozlukcu = base_bert.word_embedding_model(VocabularyManager(char_max_len).get_vocab_array("gözlükçü"))
    # prediction_scores = torch.matmul(gozluk, gozlukcu.transpose(2, 1))

    # print(prediction_scores)

    print("----- soncu ---")


    """

    for epoch in range(epochs):
        optimizer.zero_grad()
        # 1. VocabModel ile o anki iterasyon için kelime embedding'lerini hesapla
        # vocab_matrices_tensor boyutu: (vocab_size, char_max_len, bit_depth)
        # word_embeddings boyutu: (vocab_size, hidden_size)
        word_embeddings = vocab_model(vocab_matrices_tensor)
        
        # Dummy input (Örnek batch: 4 cümle, max_seq_len=32)
        # Gerçek eğitimde Tokenizer'dan gelen encode edilmiş cümleleri kullanacaksınız.
        input_ids = torch.randint(0, vocab_size, (4, max_seq_len))
        targets = input_ids.clone() 
        
        # 2. Modeli çalıştır
        # prediction_scores: (batch_size, seq_len, vocab_size)
        prediction_scores: torch.Tensor = model(input_ids, word_embeddings)
        
        # tensor_to_blender_ply(prediction_scores)
        # export_csv(prediction_scores)


        # Loss hesabı: (batch_size * seq_len, vocab_size) vs (batch_size * seq_len)
        loss = criterion(prediction_scores.view(-1, vocab_size), targets.view(-1))
        
        # 3. Geri yayılım (Backpropagation)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {loss.item():.4f}")

    print("Eğitim tamamlandı!")
    print(f"VocabModel gradyan normu: {vocab_model.linear1.weight.grad.norm().item():.4f}")

    # ----------------- MODEL KAYIT KISMI -----------------
    print("\n--- Modeller Kaydediliyor ---")
    torch.save(model.state_dict(), "bert_model.pth")
    torch.save(vocab_model.state_dict(), "vocab_model.pth")
    print("Modeller 'bert_model.pth' ve 'vocab_model.pth' olarak kaydedildi.")


    """


if __name__ == '__main__':
    test()