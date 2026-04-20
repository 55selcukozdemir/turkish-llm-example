https://yazokulu.bilimakademisi.org/yapayogrenme/2018/sunumlar/ebruarisoy-byoyo18.pdf


2️⃣ Attention Bu Noktada Ne Yapar?

Self-attention şunu yapabilir:

“gözlük” ile “göz” arasında bağ kurar

# Gelişmiş Kelime Parçalama (Tokenization) Algoritması

Bu projede oluşturduğumuz Python scripti (`app7_kelime_ayristirma.py`), Türkçenin yapısına uygun, "Byte-Pair Encoding (BPE)" mantığına benzeyen sezgisel bir kelime öğelerine ayırma algoritması kullanmaktadır.

Algoritmanın temel felsefesi kök-ek (stemming/lemmatization) bulmak değil, **"metinde yer alan kelimelerin birbiri içinde tekrar etme durumlarına bakarak, en çok tekrar eden anlamlı alt parçaları (tokenleri) bulmaktır."**

İşte adım adım algoritmamızın nasıl çalıştığı:

## Adım 1: Veri Hazırlığı (Temizleme ve Dağarcık Oluşturma)
1. **Noktalama İşaretlerinin Temizlenmesi**: Girdi olarak verilen metindeki tüm noktalama işaretleri (virgül, nokta vs.) boşluklarla değiştirilir.
2. **Boşluklardan Bölme**: Metin boşluk karakterlerinden ayrılarak geçici bir kelime havuzu oluşturulur.
3. **Eşsiz Dağarcık (Vocabulary)**: Havuzdaki tüm kelimelerin büyük/küçük harf ve Türkçe karakter (I/i) normalize edilmiş hallerine bakılarak, **sadece eşsiz kelimelerden** oluşan nihai bir referans listesi hazırlanır. Dublicate kelimeler analiz havuzunu yormamak için bu listeye alınmaz.

## Adım 2: Çapraz İçerme Analizi (İlk Parçalama)
Listenin içindeki her bir kelime, dağarcıktaki "diğer tüm kelimelere" karşı test edilir:
* Kelimenin başında veya sonunda, diğer kelime yer alıyor mu diye bakılır.
* **Örnek**: Eldeki kelime "Geliyordum" olsun. Diğer kelimeler arasında "Geliyor" kelimesi varsa algoritma bunu fark eder: `"Geliyordum" = "Geliyor" + "dum"`

### Akıllı Bölme Kuralı (Heuristic Filter)
Eğer bir kelime bölünürse, geride kalan o parça (örneğin "dum") hemen kabul edilmez. "dum" parçasının Türkçe'de anlamlı veya anlaması geçerli bir ek/kelime olup olmadığı **"metin içerisindeki diğer kelimelerde geçip geçmediğine"** bakılarak doğrulanır.

* "nefret" = "ne" + "fret" olarak bölündüğünde; "fret" kelimesi dağarcıktaki başka **hiçbir** kelimenin içinde bulunmadığı için bu kesme işlemi **tesadüfi kabul edilerek iptal edilir.**
* "Geliyordum" = "Geliyor" + "dum" olarak bölündüğünde; "dum" parçası eğer "olduğum", "koydum" vb. başka kelimelerin içinde de geçiyorsa, **bölme onaylanır.**

## Adım 3: Yinelemeli (Recursive) Alt Parçalara Bölme
Birinci adımda kelimeler kabaca parçalandıktan sonra, ortaya çıkan yeni listeye aynı işlemler **hiçbir kelimede değişiklik kalmayana dek (stabilize olana dek) defalarca** uygulanır.

Uzun kelimelerden kısaya doğru bir önceliklendirme yapılır:
1. "Alıyorum" -> "Alıyor" + "um"
2. Daha sonraki döngüde "Alıyor" -> "Al" + "ıyor" olarak parçalanıp havuza dahil edilir.
Böylece nihai olarak `"Alıyorum" = "Al" + "ıyor" + "um"` parçacıklarına kadar inilir.

## Adım 4: Sentezleme ve Çıktı Üretimi
Algoritma "artık bölünecek bir kelime kalmadığında" durur ve iki sonuç üretir:
1. **Final Token Listesi (ayristirilmis_kelimeler.csv)**: Tüm işlemlerden sonra elde edilen en küçük Türkçe alt parçacıkları ve metin içinde ne sıklıkla tekrar ettiklerini (frekanslarını) içerir.
2. **Kelime Bileşenleri (kelime_bilesenleri.csv)**: Orijinal metindeki kelimelerin, eğitilmiş bu token havuzuyla *("Hangi parçaları "+ ile yan yana koyarak bu kelimeyi tekrar elde ederim?")* şeklindeki matematiksel ispatıdır.
