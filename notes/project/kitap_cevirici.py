from PyPDF2 import PdfReader
import re

reader = PdfReader("tutunamayanlar.pdf")

tum_metin = ""

# Metni al
for page in reader.pages:
    text = page.extract_text()
    if text:
        tum_metin += text + " "

# SATIR SONLARINI TEMİZLE (en kritik kısım)
tum_metin = tum_metin.replace("\n", " ")

# Fazla boşlukları temizle
tum_metin = re.sub(r"\s+", " ", tum_metin)

# Cümlelere böl (nokta, ünlem, soru işareti sonrası)
cumleler = re.split(r'(?<=[.!?])\s+', tum_metin)

# TXT'ye yaz
with open("cikti.txt", "w", encoding="utf-8") as f:
    for cumle in cumleler:
        f.write(cumle.strip() + "\n")

print("Bitti.")