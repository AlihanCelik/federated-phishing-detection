import re
import os
import ssl
import pickle
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# macOS'ta SSL sertifika doğrulama hatası olabileceğinden bypass uygula
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def clean_text(text):
    """
    URL, HTML etiketi ve e-posta adreslerini metinden temizler.
    Sadece anlamsal içerik kalır (URL bağımsız analiz).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # URL ve linkleri kaldır
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # HTML etiketlerini temizle
    text = re.sub(r'<.*?>', '', text)

    # E-posta adreslerini temizle
    text = re.sub(r'\S+@\S+', '', text)

    # Sadece alfabetik karakterleri tut
    text = re.sub(r'[^a-z\s]', '', text)

    # Stopwords temizliği
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]

    return " ".join(cleaned_words)


def build_global_tokenizer(data_path, max_words=5000, tokenizer_save_path="tokenizer.pkl"):
    """
    Tüm veri seti üzerinde tek bir global tokenizer oluşturur ve diske kaydeder.
    Tüm istemciler bu tokenizer'ı kullanarak tutarlı kelime-indeks eşleşmesi sağlar.

    Neden gerekli:
        Her istemci kendi verisine göre ayrı tokenizer fit ederse, aynı kelime
        farklı istemcilerde farklı indekslere denk gelir. Bu durumda paylaşılan
        model ağırlıkları anlamsız hale gelir çünkü embedding katmanı farklı
        kelime-indeks haritalarıyla eğitilmiş olur.
    """
    print(f"[Global Tokenizer] '{data_path}' üzerinde oluşturuluyor...")

    df = pd.read_csv(data_path)
    texts = df['text'].astype(str).tolist()
    cleaned_texts = [clean_text(t) for t in texts]

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_texts)

    with open(tokenizer_save_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"[Global Tokenizer] {len(tokenizer.word_index)} benzersiz kelime bulundu.")
    print(f"[Global Tokenizer] Kaydedildi: {tokenizer_save_path}")
    return tokenizer


def load_global_tokenizer(tokenizer_path="tokenizer.pkl"):
    """
    Daha önce kaydedilmiş global tokenizer'ı yükler.
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer dosyası bulunamadı: '{tokenizer_path}'. "
            "Önce 'python main.py --prepare' komutuyla tokenizer oluşturun."
        )
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"[Global Tokenizer] Yüklendi: {tokenizer_path}")
    return tokenizer


def texts_to_padded(texts, tokenizer, max_length=50):
    """
    Verilen metinleri global tokenizer ile dizi haline getirir ve pad'ler.
    """
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded


def split_data_heterogeneous(texts, labels, num_clients, client_id, alpha=0.5, seed=42):
    """
    Dirichlet dağılımı kullanarak heterojen (non-IID) veri bölüşümü yapar.

    Gerçek federatif öğrenme senaryolarında her istemcinin veri dağılımı farklıdır.
    Örneğin bir kurumun e-posta trafiği çoğunlukla oltalama içerirken, başka bir
    kurumunkinde temiz e-postalar baskın olabilir. alpha parametresi bu heterojenliği
    kontrol eder: küçük alpha (0.1-0.5) yüksek heterojenlik, büyük alpha (>1) homojenliğe yaklaşır.

    Args:
        texts: Tüm metin listesi
        labels: Tüm etiket dizisi (0/1)
        num_clients: Toplam istemci sayısı
        client_id: Bu istemcinin ID'si (1'den başlar)
        alpha: Dirichlet konsantrasyon parametresi (varsayılan 0.5)
        seed: Tekrarlanabilirlik için rastgele tohum

    Returns:
        client_texts, client_labels: Bu istemciye ait veri
    """
    np.random.seed(seed)
    labels_array = np.array(labels)
    classes = np.unique(labels_array)

    client_indices = []

    for cls in classes:
        cls_indices = np.where(labels_array == cls)[0]
        np.random.shuffle(cls_indices)

        # Dirichlet dağılımından her istemci için oran üret
        proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha)

        # Oranları kümülatif indekslere çevir
        proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)
        proportions = np.clip(proportions, 0, len(cls_indices))

        # Bu istemciye düşen dilimi al
        start = proportions[client_id - 2] if client_id > 1 else 0
        end = proportions[client_id - 1]

        client_indices.extend(cls_indices[start:end].tolist())

    np.random.shuffle(client_indices)

    client_texts = [texts[i] for i in client_indices]
    client_labels = labels_array[client_indices]

    return client_texts, client_labels


def process_and_save_dataset(input_csv, output_csv, text_column, label_column):
    """
    Tüm veri setini temizler ve yeni bir dosya olarak kaydeder.
    """
    print(f"[{input_csv}] dosyası yükleniyor...")

    if not os.path.exists(input_csv):
        print(f"HATA: '{input_csv}' bulunamadı!")
        return

    df = pd.read_csv(input_csv)

    if text_column not in df.columns or label_column not in df.columns:
        print(f"HATA: '{text_column}' veya '{label_column}' sütunu bulunamadı.")
        return

    print(f"Metinler temizleniyor ({len(df)} satır)...")

    df = df.dropna(subset=[text_column, label_column])
    df[label_column] = df[label_column].astype(int)
    df['cleaned_text'] = df[text_column].apply(clean_text)

    processed_df = df[['cleaned_text', label_column]]
    processed_df = processed_df[processed_df['cleaned_text'].str.strip() != '']
    processed_df.to_csv(output_csv, index=False)

    print(f"Temizlenmiş veri seti kaydedildi: {output_csv}")
    print(f"Final satır sayısı: {len(processed_df)}")


if __name__ == "__main__":
    INPUT_PATH = "email_text.csv"
    OUTPUT_PATH = "processed_dataset.csv"
    TEXT_COL = "text"
    LABEL_COL = "label"

    process_and_save_dataset(INPUT_PATH, OUTPUT_PATH, TEXT_COL, LABEL_COL)
    build_global_tokenizer(INPUT_PATH)
