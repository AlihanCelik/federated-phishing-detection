# 🛡️ Federatif Öğrenme Tabanlı Oltalama (Phishing) Tespiti

> **Proje Ekibi:** Esra Çayırpunar, Alihan Çelik, Bedirhan Koz, Kaan Bal  
> **Danışman:** Güzin Ulutaş

---

## 📌 Proje Nedir?

Bu proje, e-posta içeriklerindeki oltalama (phishing) saldırılarını **kullanıcı verilerini dışarı çıkarmadan** tespit eden, gizlilik odaklı bir yapay zeka sistemidir.

Geleneksel sistemlerin iki temel sorunu vardır:
- **Merkezi yapay zeka modelleri:** E-posta içeriklerini merkezi sunucuya göndermek zorunda kalır → KVKK ihlali
- **Geleneksel filtreler (URL/başlık analizi):** URL kısaltma servisleri ve yeni nesil maskeleme taktikleriyle kolayca aşılır

Bu proje her iki sorunu da çözer:
- ✅ Veri yerel cihazda kalır, sadece model ağırlıkları paylaşılır
- ✅ URL'ye bakmaz, e-postanın **anlam ve bağlamını** analiz eder
- ✅ Kötü niyetli istemcilerin sistemi zehirleme girişimlerine karşı savunma içerir

---

## 🧠 Ne Kullandık?

### Katman 1 — Yerel Algılama: GloVe + BiLSTM

**GloVe (Global Vectors for Word Representation)**  
400.000 kelimeyi önceden öğrenilmiş 100 boyutlu vektörlere dönüştürür. "Acil", "şifre", "hesabınız" gibi oltalama kelimelerinin matematiksel yakınlığını yakalar.

**BiLSTM (Bidirectional Long Short-Term Memory)**  
Cümleyi hem soldan sağa hem sağdan sola okuyarak uzak kelimeler arasındaki anlamsal bağlantıları kurar. Sosyal mühendislik dilinin cümlenin geneline yayıldığını göz önünde bulundurarak URL olmadan salt metin analizi yapar.

```
"Sayın müşteri, hesabınız güvenlik nedeniyle acil kapatılacaktır."
        ← BiLSTM →
        %94 oltalama
```

---

### Katman 2 — Dağıtık İletişim: Flower Framework (Federatif Öğrenme)

Her istemci (kurum/cihaz) kendi verisinde modeli eğitir ve sadece **ağırlık güncellemelerini** sunucuya gönderir. Ham veri asla ağa çıkmaz.

```
Kurum A  ──── ağırlıklar ────►
Kurum B  ──── ağırlıklar ────►  SUNUCU → birleştirir → geri gönderir
Kurum C  ──── ağırlıklar ────►
```

**Heterojen veri dağılımı:** Dirichlet dağılımı ile her istemciye farklı sınıf oranında veri atanır. Gerçek dünya senaryosunu simüle eder (bir kurumda çok oltalama, diğerinde az).

---

### Katman 3 — Bizans Toleranslı Savunma: Krum + Kosinüs Benzerliği

#### Tehdit: Label Flipping Saldırısı
Kötü niyetli bir istemci, oltalama e-postalarını kasıtlı olarak "temiz" olarak etiketleyerek eğitir ve zehirli ağırlıkları sunucuya gönderir. Standart FedAvg bu zehirli gradyanları diğerleriyle eşit ağırlıkta birleştirir → global model çöker.

#### Savunma: Hybrid Robust Aggregation

**Krum Algoritması**  
Her istemcinin ağırlık vektörü ile diğerleri arasındaki Öklid uzaklığını hesaplar. Çoğunluktan geometrik olarak uzak olan (zehirli) vektörleri dışlar.

Güvenlik garantisi: `f < (n-2)/2`  
(f = kötü niyetli istemci sayısı, n = toplam istemci sayısı)

**Kosinüs Benzerliği**  
Vektörlerin büyüklüğüne değil yönüne bakarak ters yönde giden (zehirli) güncellemeleri tespit eder.

**Hibrit Yöntem**  
Krum + Kosinüs skorlarını %50/%50 ağırlıkla birleştirir. Her iki yöntemin zayıf noktalarını kapatır.

```
Normal istemciler  → aynı yön, birbirine yakın  ✅ seçilir
Kötü niyetli       → ters yön, uzakta           ❌ dışlanır
```

---

## 📊 Sonuçlar

20 tur (round), 6 istemci (5 normal + 1 kötü niyetli), Hybrid savunma:

| Metrik | Değer |
|--------|-------|
| En yüksek doğruluk (accuracy) | **%96.8** |
| Ortalama doğruluk | **~%85** |
| Dalgalanma (ani düşüş) | **Yok** |
| Kötü niyetli istemci tespiti | **Her round** |

---

## 📂 Proje Yapısı

```
federatifOltalamaTespiti/
│
├── main.py                    # Ana giriş noktası (--prepare / --simulate / --plot)
├── server.py                  # Sunucu + Krum/Cosine/Hybrid savunma
├── client.py                  # İstemci + Label Flipping saldırısı simülasyonu
├── run_simulation.py          # Tüm süreci başlatan orkestratör
├── plot_results.py            # Sonuç grafiği oluşturucu
├── email_text.csv             # E-posta veri seti
├── glove.6B.100d.txt          # GloVe kelime vektörleri (400K kelime, 100 boyut)
├── tokenizer.pkl              # Global tokenizer (--prepare ile oluşturulur)
├── results.json               # Simülasyon sonuçları
├── grafik_hybrid.png          # Sonuç grafiği
│
└── src/
    ├── __init__.py
    ├── bilstm_model.py        # BiLSTM model mimarisi
    ├── data_preprocessing.py  # Metin temizleme + Dirichlet veri bölüşümü + Global tokenizer
    └── glove_loader.py        # GloVe embedding matrisi yükleyici
```

---

## ⚙️ Kurulum

**1. Repoyu klonla:**
```bash
git clone https://github.com/kullanici_adi/federatifOltalamaTespiti.git
cd federatifOltalamaTespiti
```

**2. Sanal ortam oluştur ve aktif et:**
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

**3. Gerekli kütüphaneleri yükle:**
```bash
pip install -r requirements.txt
```

**4. GloVe dosyasını indir:**  
[GloVe 6B 100d](https://nlp.stanford.edu/projects/glove/) adresinden `glove.6B.100d.txt` dosyasını indirip proje kök dizinine koy.

---

## 🚀 Çalıştırma

### Adım 1 — Global tokenizer oluştur (sadece ilk seferde)
```bash
python3 main.py --prepare
```
Tüm veri seti üzerinde ortak bir tokenizer oluşturur ve `tokenizer.pkl` olarak kaydeder. Tüm istemciler bu tokenizer'ı kullanır — böylece aynı kelime her istemcide aynı indekse karşılık gelir.

---

### Adım 2 — Simülasyonu başlat
```bash
python3 main.py --simulate --num_normal 5 --num_malicious 1 --robust_method hybrid
```

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--num_normal` | Normal (güvenilir) istemci sayısı | 5 |
| `--num_malicious` | Kötü niyetli istemci sayısı | 1 |
| `--robust_method` | Savunma algoritması: `cosine` / `krum` / `hybrid` | hybrid |
| `--num_rounds` | Federatif öğrenme tur sayısı | 20 |
| `--alpha` | Dirichlet heterojenlik parametresi (küçük=heterojen) | 0.5 |

Simülasyon ~10-15 dakika sürer. Her round'un accuracy değeri terminalde canlı görünür.

---

### Adım 3 — Grafik oluştur
```bash
python3 main.py --plot
```
`grafik_hybrid.png` dosyası oluşturulur.

---

### Savunmanın çöktüğü senaryoyu test et (Krum eşiği aşıldığında)
```bash
python3 main.py --simulate --num_normal 3 --num_malicious 3 --robust_method hybrid
```
Kötü niyetli istemci sayısı eşiği aştığında (`f >= (n-2)/2`) savunmanın nasıl zayıfladığını gözlemleyebilirsin.

---

## 🛠 Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|-----------|----------------|
| Python 3.12 | Ana programlama dili |
| TensorFlow / Keras | BiLSTM model eğitimi |
| Flower (flwr) | Federatif öğrenme altyapısı |
| GloVe (Stanford NLP) | Önceden eğitilmiş kelime vektörleri |
| NLTK | Metin ön işleme, stopword temizleme |
| NumPy / Pandas | Veri işleme |
| Matplotlib | Sonuç görselleştirme |

---

## 🔬 Teknik Detaylar

**BiLSTM Mimarisi:**
```
Embedding (GloVe, 100 boyut, trainable=False)
    → BiLSTM (64 hücre: 32 ileri + 32 geri)
    → Dropout (0.5)
    → Dense (1, sigmoid)
```

**Veri Bölüşümü:**  
Her istemci verisinin %80'i eğitim, %20'si test için kullanılır. Sunucu, istemcilerin test sonuçlarını örnek sayısına göre ağırlıklı ortalama ile birleştirir.

**Krum Güvenlik Koşulu:**  
`f < (n-2)/2` — kötü niyetli istemci sayısı bu eşiğin altında kaldığı sürece global model istatistiksel kayıp yaşamaz.

---

## 📄 Lisans

MIT License
