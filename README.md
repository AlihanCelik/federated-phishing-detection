# 🛡️ Federatif Öğrenme Tabanlı Oltalama (Phishing) Tespiti

> **Proje Ekibi:** Esra Çayırpunar, Alihan Çelik, Bedirhan Koz, Kaan Bal  
> **Danışman:** Güzin Ulutaş

---

## 📌 Proje Nedir?

Bu proje, e-posta içeriklerindeki oltalama (phishing) saldırılarını **kullanıcı verilerini dışarı çıkarmadan** tespit eden, gizlilik odaklı bir yapay zeka sistemidir.

Geleneksel sistemlerin iki temel sorunu vardır:

- **Merkezi yapay zeka modelleri:** E-posta içeriklerini merkezi sunucuya göndermek zorunda kalır → KVKK/GDPR ihlali riski
- **Geleneksel filtreler (URL/başlık analizi):** URL kısaltma servisleri ve yeni nesil maskeleme taktikleriyle kolayca aşılır

Bu proje her iki sorunu da çözer:

- ✅ Ham veri yerel cihazda kalır, sadece model ağırlıkları paylaşılır
- ✅ URL'ye bakmaz, e-postanın **anlam ve bağlamını** analiz eder
- ✅ Kötü niyetli istemcilerin sistemi zehirleme girişimlerine karşı dört katmanlı savunma içerir

---

## 🏗️ Sistem Mimarisi

Sistem üç ana katmandan oluşur:

```
┌─────────────────────────────────────────────────────────┐
│  Katman 1: Yerel Algılama                               │
│  GloVe (100 boyut) + BiLSTM → Oltalama olasılığı       │
├─────────────────────────────────────────────────────────┤
│  Katman 2: Dağıtık İletişim                             │
│  Flower Framework → Sadece ağırlıklar paylaşılır        │
├─────────────────────────────────────────────────────────┤
│  Katman 3: Bizans Toleranslı Savunma                    │
│  Krum + Kosinüs + EMA + FLTrust → Zehirli gradyan tespiti│
└─────────────────────────────────────────────────────────┘
```

---

## 🧠 Teknik Bileşenler

### Katman 1 — Yerel Algılama: GloVe + BiLSTM

**GloVe (Global Vectors for Word Representation)**  
Stanford NLP tarafından 840 milyar token üzerinde eğitilmiş, 400.000 kelimeyi 100 boyutlu vektörlere dönüştüren önceden eğitilmiş bir kelime temsil modelidir. "Acil", "şifre", "hesabınız" gibi oltalama kelimelerinin matematiksel yakınlığını yakalar.

**BiLSTM (Bidirectional Long Short-Term Memory)**  
Cümleyi hem soldan sağa hem sağdan sola okuyarak uzak kelimeler arasındaki anlamsal bağlantıları kurar. Sosyal mühendislik dilinin cümlenin geneline yayıldığını göz önünde bulundurarak URL olmadan salt metin analizi yapar.

```
Model Mimarisi:
  Embedding (GloVe, 100 boyut, trainable=False)
      → BiLSTM (64 hücre: 32 ileri + 32 geri)
      → Dropout (0.5)
      → Dense (1, sigmoid)
```

**Neden GloVe donduruldu (`trainable=False`)?**  
GloVe'un önceden öğrendiği anlamsal ilişkilerin federatif eğitim sırasında bozulmasını önlemek için. Her istemci farklı veri dağılımına sahip olduğundan serbest embedding eğitimi tutarsız sonuçlar üretir.

---

### Katman 2 — Dağıtık İletişim: Flower Framework

Her istemci (kurum/cihaz) kendi verisinde modeli eğitir ve sadece **ağırlık güncellemelerini** sunucuya gönderir. Ham veri asla ağa çıkmaz.

```
Kurum A  ──── Δağırlıklar ────►
Kurum B  ──── Δağırlıklar ────►  SUNUCU → Savunma → Birleştir → Geri gönder
Kurum C  ──── Δağırlıklar ────►
```

**Heterojen (non-IID) Veri Dağılımı**  
Dirichlet dağılımı (`α` parametresi) ile her istemciye farklı sınıf oranında veri atanır. Gerçek dünya senaryosunu simüle eder: bir kurumda çok oltalama, diğerinde az.

| α değeri | Dağılım |
|----------|---------|
| 0.1 | Çok heterojen (bir istemci neredeyse tek sınıf görür) |
| 0.5 | Orta heterojen (varsayılan) |
| 5.0 | Homojen (IID'ye yakın) |

---

### Katman 3 — Bizans Toleranslı Savunma

#### Tehdit Modeli: Label Flipping Saldırısı

Kötü niyetli bir istemci, oltalama e-postalarını kasıtlı olarak "temiz" (0→1, 1→0) olarak etiketleyerek eğitir ve zehirli ağırlıkları sunucuya gönderir. Standart FedAvg bu zehirli gradyanları diğerleriyle eşit ağırlıkta birleştirir → global model bozulur.

#### Savunma Yöntemleri

Sistem dört savunma algoritması sunar:

---

**1. Krum** (`--robust_method krum`)

Her istemcinin ağırlık vektörü ile diğerleri arasındaki Öklid uzaklığını hesaplar. Çoğunluktan geometrik olarak uzak olan (zehirli) vektörleri dışlar.

```
Krum skoru(i) = Σ dist(i, j)  [en yakın n-f-2 komşu için]
```

Güvenlik garantisi: `f < (n-2)/2`  
*(f = kötü niyetli istemci sayısı, n = toplam istemci sayısı)*

> Referans: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017.

---

**2. Kosinüs Benzerliği** (`--robust_method cosine`)

Vektörlerin büyüklüğüne değil yönüne bakarak medyan güncelleme yönünden sapan (zehirli) güncellemeleri tespit eder.

```
cos(i) = (Δi · Δmedyan) / (‖Δi‖ · ‖Δmedyan‖)
```

---

**3. Hibrit** (`--robust_method hybrid`)

Krum + Kosinüs + EMA (Exponential Moving Average) güven skorlarını birleştirir.

```
Anlık skor  = 0.35 × Krum_norm + 0.35 × Kosinüs_norm + 0.30 × Tutarsızlık_norm
Nihai skor  = 0.35 × anlık + 0.65 × EMA  (≥3 round geçmişi varsa)
Birleşik    = 0.50 × nihai + 0.50 × EMA
```

**EMA (Exponential Moving Average) Güven Skoru**  
Her istemcinin geçmiş round performansını üstel ağırlıkla takip eder. Tek bir "temiz" round kötü geçmişi silemez; sürekli şüpheli davranan istemciler zamanla düşük EMA skoruna sahip olur.

---

**4. FLTrust** (`--robust_method fltrust`) ← *Varsayılan ve önerilen*

Sunucunun küçük, temiz ve doğrulanmış bir referans veri setine sahip olduğu varsayımına dayanır. Her round sunucu bu referans veri üzerinde bir adım gradient hesaplar ve her istemcinin güncellemesini bu referans gradyanla karşılaştırır.

```
FLTrust skoru(i) = ReLU( cos(Δi, Δreferans) )
```

ReLU sayesinde referans gradyanla tamamen ters yönde giden güncellemeler sıfır güven alır.

> Referans: Wang et al., "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping", NDSS 2021.

**Gerçek dünya uyumu:** Bir kurumun IT/güvenlik ekibinin elle etiketlenmiş küçük bir e-posta havuzuna sahip olması makul bir varsayımdır. Bu veri istemcilere paylaşılmaz, yalnızca sunucuda kalır.

---

## 📊 Sonuçlar

### Hibrit Savunma — 20 Tur, 6 İstemci (5 Normal + 1 Kötü Niyetli)

| Metrik | Değer |
|--------|-------|
| En yüksek doğruluk | **%98.34** |
| Ortalama doğruluk | **%98.05** |
| Son tur doğruluğu | **%96.79** |
| Doğru tespit (round) | **17 / 20** |
| Yanlış seçim (round) | **3 / 20** |
| Tespit başarı oranı | **%85.00** |

> İlk 3 roundda kaçırma yaşanır çünkü EMA ve tutarsızlık skoru için yeterli geçmiş birikmemiştir. Round 4'ten itibaren savunma tam kapasitede çalışır.

### Krum Güvenlik Sınırı

| Senaryo | Koşul | Durum |
|---------|-------|-------|
| 6 istemci, 1 kötü niyetli | 1 < 2.0 | ✅ Güvenli |
| 6 istemci, 2 kötü niyetli | 2 < 2.0 | ⚠️ Sınırda |
| 4 istemci, 2 kötü niyetli | 2 < 1.0 | ❌ Güvensiz |

---

## 📂 Proje Yapısı

```
federatifOltalamaTespiti/
│
├── main.py                    # Ana giriş noktası (--prepare / --simulate / --plot)
├── server.py                  # Sunucu + Krum/Cosine/Hybrid/FLTrust savunma
├── client.py                  # İstemci + Label Flipping saldırısı simülasyonu
├── run_simulation.py          # Tüm süreci başlatan orkestratör
├── plot_results.py            # Sonuç grafiği oluşturucu
│
├── email_text.csv             # E-posta veri seti (~53.000 örnek)
├── glove.6B.100d.txt          # GloVe kelime vektörleri (400K kelime, 100 boyut)
├── tokenizer.pkl              # Global tokenizer (--prepare ile oluşturulur)
├── results.json               # Simülasyon sonuçları (otomatik oluşturulur)
├── grafik_<method>.png        # Sonuç grafiği (otomatik oluşturulur)
├── requirements.txt           # Python bağımlılıkları
│
└── src/
    ├── __init__.py
    ├── bilstm_model.py        # BiLSTM model mimarisi
    ├── data_preprocessing.py  # Metin temizleme, Dirichlet bölüşümü, global tokenizer
    └── glove_loader.py        # GloVe embedding matrisi yükleyici
```

---

## ⚙️ Kurulum

### Gereksinimler

- Python 3.10+
- macOS / Linux (Windows'ta `source` yerine `venv\Scripts\activate` kullanın)
- ~2 GB disk alanı (GloVe dosyası için)

### Adımlar

**1. Repoyu klonla:**
```bash
git clone https://github.com/AlihanCelik/federatifOltalamaTespiti.git
cd federatifOltalamaTespiti
```

**2. Sanal ortam oluştur ve aktif et:**
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

**3. Bağımlılıkları yükle:**
```bash
pip install -r requirements.txt
```

**4. GloVe dosyasını indir:**

[Stanford NLP GloVe sayfasından](https://nlp.stanford.edu/projects/glove/) `glove.6B.zip` dosyasını indirip içindeki `glove.6B.100d.txt` dosyasını proje kök dizinine koy.

```bash
# Alternatif: wget ile doğrudan indir
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
```

---

## 🚀 Çalıştırma

### Adım 1 — Global tokenizer oluştur *(sadece ilk seferde)*

```bash
python3 main.py --prepare
```

Tüm veri seti üzerinde ortak bir tokenizer oluşturur ve `tokenizer.pkl` olarak kaydeder. Tüm istemciler bu tokenizer'ı kullanır — böylece aynı kelime her istemcide aynı indekse karşılık gelir.

---

### Adım 2 — Simülasyonu başlat

```bash
python3 main.py --simulate
```

Varsayılan ayarlarla çalışır: 5 normal + 1 kötü niyetli istemci, FLTrust savunması, 50 tur.

**Tüm parametreler:**

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--num_normal` | Normal (güvenilir) istemci sayısı | `5` |
| `--num_malicious` | Kötü niyetli istemci sayısı | `1` |
| `--robust_method` | Savunma algoritması: `cosine` / `krum` / `hybrid` / `fltrust` | `fltrust` |
| `--num_rounds` | Federatif öğrenme tur sayısı | `50` |
| `--alpha` | Dirichlet heterojenlik parametresi | `0.5` |
| `--server_data_size` | FLTrust referans veri seti boyutu | `100` |
| `--data_path` | CSV veri seti yolu | `email_text.csv` |
| `--glove_path` | GloVe dosyası yolu | `glove.6B.100d.txt` |

---

### Adım 3 — Grafik oluştur

```bash
python3 main.py --plot
```

`grafik_<method>.png` dosyası oluşturulur. Grafik şunları içerir:
- Her round'un accuracy ve loss değerleri
- Ortalama accuracy çizgisi
- Savunmanın zorlandığı roundların vurgulanması
- Label flipping tespit istatistikleri tablosu

---

### Örnek Senaryolar

**Savunma yöntemlerini karşılaştır:**
```bash
# FLTrust (önerilen)
python3 main.py --simulate --robust_method fltrust --num_normal 5 --num_malicious 1

# Hibrit
python3 main.py --simulate --robust_method hybrid --num_normal 5 --num_malicious 1

# Sadece Krum
python3 main.py --simulate --robust_method krum --num_normal 5 --num_malicious 1
```

**Krum güvenlik sınırını test et (savunma zorlanır):**
```bash
python3 main.py --simulate --num_normal 3 --num_malicious 2 --robust_method hybrid
```

**Yüksek heterojenlik senaryosu:**
```bash
python3 main.py --simulate --alpha 0.1 --num_normal 5 --num_malicious 1
```

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| Python | 3.10+ | Ana programlama dili |
| TensorFlow / Keras | 2.21.0 | BiLSTM model eğitimi |
| Flower (flwr) | 1.29.0 | Federatif öğrenme altyapısı |
| GloVe (Stanford NLP) | 6B.100d | Önceden eğitilmiş kelime vektörleri |
| NLTK | 3.9.4 | Metin ön işleme, stopword temizleme |
| NumPy / Pandas | — | Veri işleme |
| Matplotlib | — | Sonuç görselleştirme |

---

## 🔬 Algoritma Referansları

| Algoritma | Kaynak |
|-----------|--------|
| Krum | Blanchard et al., *"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"*, NeurIPS 2017 |
| FLTrust | Wang et al., *"FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping"*, NDSS 2021 |
| Dirichlet non-IID | Hsieh et al., *"Quagmire of Benchmarks"*, 2020 |
| GloVe | Pennington et al., *"GloVe: Global Vectors for Word Representation"*, EMNLP 2014 |

---

## ❓ Sık Sorulan Sorular

**S: Simülasyon ne kadar sürer?**  
A: Donanıma bağlı olarak 20 turda ~10-20 dakika, 50 turda ~25-45 dakika sürer. GPU yoksa CPU üzerinde çalışır.

**S: `tokenizer.pkl` zaten varsa `--prepare` tekrar çalıştırmam gerekir mi?**  
A: Hayır. Veri seti değişmedikçe mevcut tokenizer kullanılmaya devam eder.

**S: Kötü niyetli istemci sayısını artırırsam ne olur?**  
A: Krum güvenlik koşulu `f < (n-2)/2` aşılırsa savunma zayıflar. Örneğin 6 istemcide 2'den fazla kötü niyetli istemci bu eşiği aşar.

**S: FLTrust neden varsayılan?**  
A: Sunucunun küçük bir referans veri setine sahip olduğu varsayımı bu senaryo için gerçekçidir ve diğer yöntemlere kıyasla daha kararlı tespit sağlar.

---

## 📄 Lisans

MIT License
