from tokenizer import Tokenizer
from torch.utils.data import Dataset
from karbon_turk_config import KarbonTurkConfig
import numpy as np

class DataPreparer(Dataset):
    def __init__(self, config: KarbonTurkConfig):
        self.config = config
        
        satirlar = []

        with open(config.data_set, "r", encoding="utf-8") as file:
            for satir in file:
                satirlar.append(satir.strip())

        self.satirlar = satirlar

        self.bos_token = self.get_word_representing(config.spacial_tokens_BOS)
        self.eos_token = self.get_word_representing(config.spacial_tokens_EOS)

        self.padding_token = self.get_word_representing(config.spacial_tokens_padding)
    
    def __len__(self):
        return len(self.satirlar)
    
    def __getitem__(self, idx):
        ret_val_txt = self.satirlar[idx]
        tokens = Tokenizer.tokenize(ret_val_txt)
        
        # Kelime temsilcilerini (vektör/id) listeye çeviriyoruz
        input_tokens = [self.get_word_representing(token) for token in tokens]
        
        # BOS ve EOS ekleneceği için gerçek veri uzunluğu = len(input_tokens) + 2
        # Maksimum uzunluktan bu değeri çıkararak gereken padding sayısını buluyoruz
        num_padding_token = self.config.max_seq_len - len(input_tokens) - 2
        
        if num_padding_token < 0:
            # Eğer cümle çok uzunsa, hata fırlatmak yerine kırpabilirsiniz (truncation)
            # Ya da bu şekilde kalabilir:
            raise ValueError(f"Sentence is too long ({len(input_tokens)} tokens). Max allowed is {self.config.max_seq_len - 2}")
        
        # Padding listesini oluşturuyoruz
        padding_tokens = [self.padding_token] * num_padding_token
        
        try: 
            # Standart sıralama: [BOS] + INPUT + [EOS] + [PADDING]
            # Liste olarak birleştirip tek seferde numpy array veya torch tensörüne çevirmek en güvenlisidir
            full_sequence = [self.bos_token] + input_tokens + [self.eos_token] + padding_tokens
            
            # Tüm elemanların numpy array formatında ve aynı veri tipinde olduğundan emin olalım
            ret_val = np.array(full_sequence, dtype=np.float32) # Veya embedding id ise np.int64
            
        except Exception as e:
            self.config.logger.error(f"Array birleştirme hatası: {e}")
            ret_val = np.array([])

        return {
            "input": ret_val,
            # NOT: Eğer hata devam ederse, test sırasında "input_txt" anahtarını 
            # döndürmeyi bırakın. PyTorch DataLoader bazen ham string listelerini batch yapamaz.
            "input_txt": ret_val_txt 
        }
    def get_word_representing(self, word_txt):
        return self.config.vacobulary_manager.get_vocab_array(word_txt)

        