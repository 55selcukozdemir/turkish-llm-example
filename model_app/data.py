from tokenizer import Tokenizer
from torch.utils.data import Dataset
from karbon_turk_config import KarbonTurkConfig
import numpy as np

class DataPreparer(Dataset):
    def __init__(self, config: KarbonTurkConfig):
        self.config = config
        
        satirlar = []
        number_continued_line = 0
        with open(config.data_set, "r", encoding="utf-8") as file:
            for satir in file:
                
                # temporarliy
                str_line = satir.strip()
                if config.max_seq_len < (len(Tokenizer.tokenize(str_line)) - 3):
                    number_continued_line += 1
                    continue
                satirlar.append(satir.strip())
        config.logger.info(F"umber of continiued line {number_continued_line}")
        self.satirlar = satirlar

        self.bos_token = self.get_word_representing(config.spacial_tokens_BOS)
        self.eos_token = self.get_word_representing(config.spacial_tokens_EOS)

        self.padding_token = self.get_word_representing(config.spacial_tokens_padding)
        
        # check dataset
        
        self.check_max_squence_length()
    def check_max_squence_length(self):
        max_len = 0
        max_len_line = 0
        for index, i in enumerate(self.satirlar):
            tokens_of_line_lenght = len(Tokenizer.tokenize(i))
            
            if  tokens_of_line_lenght > max_len:
                max_len = tokens_of_line_lenght
                max_len_line = index
                
        if self.config.max_seq_len < max_len - 2:
            raise ValueError(F"The maximum number of sentence has been exceeded({max_len} - {max_len_line})")
        return max_len    
            
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

        