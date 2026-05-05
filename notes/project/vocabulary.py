import numpy as np
import torch

class Vocabulary():
    def __init__(self, max_seq_len, bit_depth=8):
        """
            ['<PAD>',   # padding (sequence equalization)
            '<UNK>',   # bilinmeyen karakter
            '<BOS>',   # beginning of sentence
            '<EOS>',   # end of sentence
            '<MASK>',  # masking (BERT tarzı)
            ' ']      
        """
        self.vocab = (
            ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>', ' '] +
            
            list('abcçdefgğhıijklmnoöprsştuüvyz') +
            list('ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ') +
            list('0123456789') +
            
            ['.',',',';',':','?','!','"',"'",'(',')','[',']','{','}','-','–','—','…','/','\\','|'] +
            
            ['\n']
        )

        self.char_dic = {}
        for i in range(0, len(self.vocab)):
            bits = [int(b) for b in format(i, '08b')]
            self.char_dic[self.vocab[i]] = bits

        self.max_seq_len = max_seq_len
        self.bit_depth = bit_depth

    def __get_angle(self, pos, i, d_model):
        """
        Pozisyon ve boyut indeksine göre açı değerini hesaplar
        """
        return pos / (10 ** ((2 * (i // 2)) / d_model))

    def __positional_encoding_matrix(self):
        """
        Tüm positional encoding matrisini oluşturur
        """
        PE = np.ones((self.max_seq_len, self.bit_depth))

        for pos in range(self.max_seq_len):
            for i in range(self.bit_depth):
                angle = self.__get_angle(pos, i, self.bit_depth)
                if i % 2 == 0:
                    PE[pos, i] = np.sin(angle)
                else:
                    PE[pos, i] = np.cos(angle)

        return PE

    def __8bit_encode(self, text):
        """
        return: (max_seq_len, bit_depth) 8-bit matrix
        """
        result = np.zeros((self.max_seq_len, self.bit_depth), dtype=np.float32)

        for i, ch in enumerate(text):
            if i >= self.max_seq_len:
                break
            
            if ch in self.char_dic:
                bits = self.char_dic[ch]
            else:
                bits = self.char_dic['<UNK>']

            result[i] = bits

        return result

    def get_vocab_array(self, vocab_txt):
        """
        Kelimeyi alır, 8-bitlik matrise dönüştürür ve positional encoding ekler.
        """
        encoded = self.__8bit_encode(vocab_txt)
        pe = self.__positional_encoding_matrix()
        
        # Sadece karakter olan kısımlara veya tamamına PE eklenebilir. 
        # Şimdilik tamamına ekliyoruz.
        output = encoded + pe
        return output
