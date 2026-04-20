# Kelime Düzeyinde Vektörleştirme

## Amaç

Her kelime için **512 boyutlu tek bir vektör** üretmek. Bu vektör, kelimenin harflerinin hem kimliğini hem de kelime içindeki konumunu kodlar.

## Nasıl Çalışır?

```
kelime_vektörü = ortalama( char_embedding(harf) × pos_embedding(pozisyon) )
```

### Adımlar

1. **Karakter Embedding**: Her harf için 512 boyutlu benzersiz bir vektör (`nn.Embedding`)
2. **Pozisyon Embedding**: Her pozisyon (0, 1, 2, ...) için 512 boyutlu bir çarpan vektörü (`nn.Embedding`)
3. **Element-wise Çarpım**: `harf_vektörü × pozisyon_vektörü` → harfin kimliği + konumu tek vektörde
4. **Ortalama**: Kelimedeki tüm harf vektörlerinin ortalaması → **kelime vektörü**

### Örnek: "adam" kelimesi

```
a(pos=0): char_emb('a') × pos_emb(0) = [0.0681, 0.0239, -0.0079, ...]
d(pos=1): char_emb('d') × pos_emb(1) = [-0.0390, 0.0575, 0.0213, ...]
a(pos=2): char_emb('a') × pos_emb(2) = [0.1010, 0.0239, -0.0074, ...]
m(pos=3): char_emb('m') × pos_emb(3) = [-0.0185, 0.0441, 0.0163, ...]

kelime_vektörü = ortalama(4 vektör) → [0.0279, 0.0373, 0.0056, ...]  (512 boyut)
```

> Aynı harf ('a') farklı pozisyonlarda farklı vektörler üretir çünkü pozisyon çarpanı değişir.

---

## Model Detayları

| Parametre | Değer |
|---|---|
| Embedding boyutu | 512 |
| Max kelime uzunluğu | 64 |
| Karakter sözlük boyutu | 110 |
| Toplam parametre | 89,088 |

---

## Çalıştırma

```bash
/opt/miniconda3/envs/pytorch/bin/python app3_kelime_vektorleri.py
```

---

## Sonuçlar

### Kelime Vektörleri

| Kelime | Vektör (ilk 5 değer) | Norm |
|---|---|---|
| adam | [0.0279, 0.0373, 0.0056, 0.0108, 0.0004] | 0.6525 |
| dama | [0.0303, 0.0380, 0.0071, 0.0133, 0.0031] | 0.6518 |
| merhaba | [0.0011, 0.0047, 0.0033, 0.0181, 0.0132] | 0.4546 |
| dünya | [0.0081, 0.0275, 0.0052, -0.0240, -0.0034] | 0.4889 |
| bilgisayar | [0.0099, 0.0009, -0.0170, 0.0424, 0.0392] | 0.4265 |
| bilgi | [0.0035, 0.0148, -0.0243, 0.0855, 0.0311] | 0.5926 |
| gel | [-0.0112, 0.0437, -0.0085, 0.0870, 0.0364] | 0.6662 |
| geldi | [-0.0141, 0.0354, -0.0063, 0.0668, 0.0357] | 0.5053 |

### Kosinüs Benzerlik Sonuçları

| Kelime Çifti | Kosinüs Benzerliği | Öklid Mesafesi | Açıklama |
|---|---|---|---|
| bilgi ↔ bilgisayar | **0.7243** | 0.4086 | Ortak kök paylaşıyor |
| gel ↔ geldi | **0.7789** | 0.4180 | Aynı kökten türemiş |
| adam ↔ dama | **0.9924** | 0.0802 | Aynı harfler, farklı sıra |
| yapay ↔ zeka | 0.2168 | 0.7441 | Farklı kelimeler |
| anne ↔ baba | 0.2210 | 0.9102 | Farklı kelimeler |

### Sıralama Farkı: "adam" vs "dama"

İki kelime aynı harflerden (a, d, a, m) oluşur ama farklı sırada dizilmiştir. Pozisyon çarpanı sayesinde farklı vektörler üretilir:

```
"adam":                          "dama":
  a (pos=0): [0.0681, ...]        d (pos=0): [-0.0285, ...]
  d (pos=1): [-0.0390, ...]       a (pos=1): [0.0931, ...]
  a (pos=2): [0.1010, ...]        m (pos=2): [-0.0233, ...]
  m (pos=3): [-0.0185, ...]       a (pos=3): [0.0800, ...]
```

### Kosinüs Benzerlik Matrisi

```
              adam    dama  merhaba  dünya   anne   baba  yapay   zeka  bilgisayar  bilgi   gel   geldi
   adam     │ 1.00   0.99   0.63    0.51   0.27   0.57   0.48   0.31    0.41      0.02  -0.04   0.15
   dama     │ 0.99   1.00   0.64    0.50   0.26   0.58   0.49   0.30    0.41      0.02  -0.03   0.15
   merhaba  │ 0.63   0.64   1.00    0.20   0.29   0.66   0.36   0.42    0.50      0.14   0.21   0.15
   dünya    │ 0.51   0.50   0.20    1.00   0.52   0.25   0.52   0.11    0.28     -0.04  -0.06   0.14
   anne     │ 0.27   0.26   0.29    0.52   1.00   0.22   0.17   0.34    0.09     -0.07   0.16   0.10
   baba     │ 0.57   0.58   0.66    0.25   0.22   1.00   0.42   0.27    0.47      0.19  -0.04  -0.06
   yapay    │ 0.48   0.49   0.36    0.52   0.17   0.42   1.00   0.22    0.49     -0.02  -0.06  -0.07
   zeka     │ 0.31   0.30   0.42    0.11   0.34   0.27   0.22   1.00    0.19      0.00   0.31   0.21
   bilgisayar│ 0.41   0.41   0.50    0.28   0.09   0.47   0.49   0.19    1.00      0.72   0.29   0.48
   bilgi    │ 0.02   0.02   0.14   -0.04  -0.07   0.19  -0.02   0.00    0.72      1.00   0.45   0.70
   gel      │-0.04  -0.03   0.21   -0.06   0.16  -0.04  -0.06   0.31    0.29      0.45   1.00   0.78
   geldi    │ 0.15   0.15   0.15    0.14   0.10  -0.06  -0.07   0.21    0.48      0.70   0.78   1.00
```

### Toplu Vektörleştirme

Metin: *"Yapay zeka ve derin öğrenme dünyayı değiştiriyor"*

| # | Kelime | Vektör (ilk 5 değer) | Norm |
|---|---|---|---|
| 1 | Yapay | [0.0339, 0.0187, -0.0107, -0.0043, -0.0038] | 0.5527 |
| 2 | zeka | [0.0211, 0.0171, 0.0098, -0.0028, 0.0266] | 0.5355 |
| 3 | ve | [-0.0238, 0.0196, -0.0262, 0.0315, 0.0798] | 0.7817 |
| 4 | derin | [-0.0314, -0.0010, 0.0213, 0.0051, 0.0184] | 0.5059 |
| 5 | öğrenme | [-0.0184, -0.0028, 0.0163, 0.0284, -0.0083] | 0.4607 |
| 6 | dünyayı | [0.0147, 0.0155, -0.0088, -0.0064, -0.0097] | 0.4793 |
| 7 | değiştiriyor | [-0.0027, -0.0087, -0.0089, 0.0284, 0.0237] | 0.4389 |

---

## Önemli Gözlemler

- ✅ **Ortak köklü kelimeler** (bilgi/bilgisayar: 0.72, gel/geldi: 0.78) doğal olarak yüksek benzerliğe sahip
- ✅ **Pozisyon çarpanı** sayesinde harf sırası önemli: "adam" ≠ "dama" (ancak aynı harflerden oluştukları için yine de yüksek benzerlik: 0.99)
- ✅ **Farklı kelimeler** düşük benzerliğe sahip (yapay ↔ zeka: 0.22, anne ↔ baba: 0.22)
- ⚠️ Model **eğitilmemiş** (rastgele ağırlıklarla). Eğitim sonrası kelime vektörleri anlamsal olarak daha tutarlı hale gelir
