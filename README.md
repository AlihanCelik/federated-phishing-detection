# Federatif Öğrenme Tabanlı Oltalama (Phishing) Tespiti

## 📌 Proje Hakkında
Projenin temel amacı; kullanıcı verilerinin gizliliğini ihlal etmeden (KVKK uyumlu), e-posta gövdelerindeki sosyal mühendislik taktiklerini ve anlamsal örüntüleri analiz ederek tespit eden bir savunma sistemi geliştirmektir. Geleneksel merkezi modellerin aksine, **Federatif Öğrenme (Federated Learning)** mimarisi kullanılarak veriler yerel cihazlarda tutulur. Ayrıca sistem, kötü niyetli istemcilerin gerçekleştirebileceği **Label Flipping** (etiket çevirme) saldırılarına karşı **Robust Aggregation** algoritmaları ile güçlendirilmiştir.

## 🚀 Temel Özellikler
* **Gizlilik Odaklı:** Veriler asla cihaz dışına çıkmaz, sadece model ağırlıkları sunucuyla paylaşılır.
* **Anlamsal Analiz:** Metin içi dizisel ilişkileri yakalamak için **BiLSTM** mimarisi kullanılır.
* **Güvenilir Yapay Zeka (Trustworthy AI):** Krum ve Kosinüs Benzerliği tabanlı savunma mekanizmaları ile Bizans toleranslı bir yapı sunar.
* **Sıfır Gün (Zero-Day) Koruması:** URL analizi yerine metin bağlamına odaklanarak yeni nesil oltalama tekniklerine karşı direnç sağlar.

## 🛠 Kullanılan Teknolojiler
* **Dil:** Python
* **Derin Öğrenme:** TensorFlow / Keras (BiLSTM)
* **Federatif Öğrenme:** Flower (flwr)
* **NLP:** NLTK, GloVe (Word Embeddings)
* **Veri Analizi:** Pandas, NumPy

## 📂 Proje Yapısı
```text
Federatif-Oltalama-Tespiti/
│
├── data/                     # Oltalama veri setleri (.csv)
├── models/                   # GloVe vektörleri ve kayıtlı modeller
├── src/                      # Kaynak kod modülleri
│   ├── data_preprocessing.py # Metin temizleme ve NLP işlemleri
│   ├── glove_loader.py       # Kelime vektörü yükleme
│   └── bilstm_model.py       # BiLSTM sinir ağı mimarisi
├── main.py                   # Merkezi eğitim (baseline) ana dosyası
├── server.py                 # Federatif öğrenme sunucu kodu (FL)
├── client.py                 # Federatif öğrenme istemci kodu (FL)
├── requirements.txt          # Gerekli kütüphaneler listesi
└── README.md                 # Proje dokümantasyonu
```

## ⚙️ Kurulum
* Projeyi klonlayın.

* Sanal ortamınızı oluşturun ve aktif edin:
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

* Gerekli kütüphaneleri yükleyin:
```
pip install -r requirements.txt
```