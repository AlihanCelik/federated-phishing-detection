import argparse
import os
import numpy as np
import pandas as pd
import flwr as fl

from src.bilstm_model import build_bilstm_model
from src.glove_loader import load_glove_embeddings
from src.data_preprocessing import (
    load_global_tokenizer,
    texts_to_padded,
    split_data_heterogeneous,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PhishingClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test,
                 is_malicious=False, client_id=0):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train.copy()
        self.x_test  = x_test
        self.y_test  = y_test.copy()   # test seti HİÇBİR ZAMAN flip edilmez
        self.is_malicious = is_malicious
        self.client_id = client_id

        # Kaç örneğin flip edildiğini kaydet — sunucuya metadata olarak gönderilir
        self.flipped_count = 0
        self.total_train   = len(self.y_train)

        if self.is_malicious:
            # Tüm etiketler ters çevrilir (0→1, 1→0).
            # Sadece label=1'i flip etmek, az phishing örneği olan partition'larda
            # çok zayıf bir saldırı üretir ve tespit edilemez hale gelir.
            # Tam flip daha güçlü bir saldırı sinyali üretir → savunma test edilebilir.
            original = self.y_train.copy()
            self.y_train = 1 - self.y_train
            self.flipped_count = int(np.sum(original != self.y_train))
            print(
                f"[KÖTÜ NİYETLİ İSTEMCİ #{self.client_id}] Label Flipping uygulandı: "
                f"{self.flipped_count}/{self.total_train} örnek flip edildi "
                f"(%{self.flipped_count/max(self.total_train,1)*100:.1f})"
            )

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print(f"\n--- [İstemci #{self.client_id}] Sunucudan ağırlıklar alındı, eğitim başlıyor ---")
        self.model.set_weights(parameters)

        if self.is_malicious:
            print(f"[KÖTÜ NİYETLİ #{self.client_id}] Zehirlenmiş etiketlerle eğitim yapılıyor "
                  f"({self.flipped_count} flip'li örnek).")

        self.model.fit(self.x_train, self.y_train, epochs=2, batch_size=32, verbose=1)

        # Sunucuya gönderilen metadata:
        #   is_malicious  → sunucu bu bilgiyi doğrulama için kullanır (gerçekte bilinmez,
        #                   burada tespit başarısını ölçmek için simülasyon amaçlı gönderilir)
        #   flipped_count → kaç örnek zehirlendi
        #   total_train   → toplam eğitim örneği
        metrics = {
            "is_malicious":  int(self.is_malicious),
            "flipped_count": self.flipped_count,
            "total_train":   self.total_train,
            "client_id":     self.client_id,
        }
        return self.model.get_weights(), len(self.x_train), metrics

    def evaluate(self, parameters, config):
        print(f"\n--- [İstemci #{self.client_id}] Değerlendirme başlıyor ---")
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def load_data(data_path):
    """CSV'den ham metin ve etiketleri yükler."""
    if data_path and os.path.exists(data_path):
        print(f"Veri '{data_path}' konumundan yükleniyor...")
        df = pd.read_csv(data_path)
        texts = df['text'].astype(str).tolist()
        labels = np.array(df['label'].astype(int).tolist())
    else:
        print(f"UYARI: '{data_path}' bulunamadı! Mock veri üretiliyor...")
        n = 1000
        texts = [
            "Dear customer, your account will be suspended urgently click here"
            if i % 2 == 0
            else "Meeting scheduled for tomorrow at 10am please confirm attendance"
            for i in range(n)
        ]
        labels = np.array([1 if i % 2 == 0 else 0 for i in range(n)])

    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Federatif Öğrenme İstemcisi (BiLSTM)")
    parser.add_argument("--client_id", type=int, default=1, help="İstemci Kimliği (1'den başlar)")
    parser.add_argument("--malicious", action="store_true", help="Kötü niyetli istemci (label flipping saldırısı)")
    parser.add_argument("--data_path", type=str, default="email_text.csv", help="CSV veri seti yolu")
    parser.add_argument("--num_clients", type=int, default=3, help="Toplam istemci sayısı")
    parser.add_argument("--glove_path", type=str, default="glove.6B.100d.txt", help="GloVe dosyası yolu")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.pkl", help="Global tokenizer dosyası yolu")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet heterojenlik parametresi (küçük=heterojen, büyük=homojen)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"İstemci #{args.client_id} başlatılıyor... (Kötü Niyetli: {args.malicious})")
    print(f"{'='*60}")

    # --- 1. Veriyi yükle ---
    texts, labels = load_data(args.data_path)

    # --- 2. Heterojen (non-IID) veri bölüşümü ---
    # Dirichlet dağılımı ile her istemci farklı sınıf dağılımına sahip olur.
    client_texts, client_labels = split_data_heterogeneous(
        texts, labels,
        num_clients=args.num_clients,
        client_id=args.client_id,
        alpha=args.alpha,
        seed=42
    )
    print(f"İstemci #{args.client_id}: {len(client_texts)} örnek | "
          f"Oltalama oranı: %{np.mean(client_labels)*100:.1f}")

    # --- 3. Global tokenizer yükle (tüm istemciler aynı tokenizer'ı kullanır) ---
    # Bu, farklı istemcilerde aynı kelimenin aynı indekse karşılık gelmesini garantiler.
    max_words = 5000
    max_length = 50

    tokenizer = load_global_tokenizer(args.tokenizer_path)
    word_index = tokenizer.word_index

    x_padded = texts_to_padded(client_texts, tokenizer, max_length=max_length)

    # --- 4. Train/test bölümü ---
    split_idx = int(len(x_padded) * 0.8)
    x_train, y_train = x_padded[:split_idx], client_labels[:split_idx]
    x_test, y_test = x_padded[split_idx:], client_labels[split_idx:]

    vocab_size = min(max_words, len(word_index) + 1)

    # --- 5. GloVe embedding matrisini yükle ---
    embedding_matrix = None
    if os.path.exists(args.glove_path):
        embedding_matrix = load_glove_embeddings(
            args.glove_path, word_index, vocab_size, embedding_dim=100
        )
    else:
        print(f"UYARI: GloVe dosyası ({args.glove_path}) bulunamadı. "
              "Model kelimeleri sıfırdan öğrenecek.")

    # --- 6. Modeli oluştur ---
    model = build_bilstm_model(
        vocab_size=vocab_size,
        embedding_dim=100,
        max_length=max_length,
        embedding_matrix=embedding_matrix
    )

    # --- 7. Flower istemcisini başlat ---
    client = PhishingClient(
        model, x_train, y_train, x_test, y_test,
        is_malicious=args.malicious,
        client_id=args.client_id,
    )

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )


if __name__ == "__main__":
    main()
