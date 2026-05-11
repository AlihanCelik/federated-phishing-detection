import argparse
import os
import numpy as np
import pandas as pd
import flwr as fl

from src.bilstm_model import build_bilstm_model, tokenize_and_pad_texts
from src.glove_loader import load_glove_embeddings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PhishingClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, is_malicious=False):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.is_malicious = is_malicious

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print("\n--- Sunucudan ağırlıklar alındı, yerel eğitim başlıyor ---")
        self.model.set_weights(parameters)

        # label flipping saldırısı
        if self.is_malicious:
            print("Bu istemci KÖTÜ NİYETLİ olarak yapılandırıldı.")
            print("         Label Flipping uygulanıyor: Tüm oltalama e-postaları 'temiz' olarak işaretleniyor.")
            self.y_train = np.where(self.y_train == 1, 0, self.y_train)

        # eğitimi başlat
        self.model.fit(self.x_train, self.y_train, epochs=2, batch_size=32, verbose=1)
        
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        print("\n--- Değerlendirme başlıyor ---")
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def load_or_generate_data(data_path=None, num_samples=1000):
  
    if data_path and os.path.exists(data_path):
        print(f"Veri '{data_path}' konumundan yükleniyor...")
        df = pd.read_csv(data_path)
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
    else:
        print(f"UYARI: '{data_path}' bulunamadı! Simülasyonun çalışabilmesi için {num_samples} adet rastgele (mock) veri üretiliyor...")
        texts = ["Bu rastgele oluşturulmuş bir test e-postasıdır." if i % 2 == 0 else "Lütfen şifrenizi sıfırlamak için bu oltalama linkine tıklayın!" for i in range(num_samples)]
        labels = np.array([0 if i % 2 == 0 else 1 for i in range(num_samples)])
        
    return texts, np.array(labels)

def main():
    parser = argparse.ArgumentParser(description="Federatif Öğrenme İstemcisi (BiLSTM)")
    parser.add_argument("--client_id", type=int, default=1, help="İstemci Kimliği (ID)")
    parser.add_argument("--malicious", action="store_true", help="Bu istemciyi zehirleme saldırısı için kötü niyetli yap")
    parser.add_argument("--data_path", type=str, default="", help="CSV formatındaki veri setinin yolu")
    parser.add_argument("--num_clients", type=int, default=3, help="Sistemdeki toplam istemci sayısı")
    parser.add_argument("--glove_path", type=str, default="glove.6B.100d.txt", help="GloVe kelime vektörleri dosyasının yolu")
    args = parser.parse_args()

    print(f"İstemci #{args.client_id} başlatılıyor... (Kötü Niyetli: {args.malicious})")

    texts, labels = load_or_generate_data(args.data_path)
    if texts is None or labels is None:
        return
        
    # veri setini bölüştür
    np.random.seed(42) 
    indices = np.arange(len(texts))
    np.random.shuffle(indices)
    
    texts = [texts[i] for i in indices]
    labels = labels[indices]
    
    chunk_size = len(texts) // args.num_clients
    start_idx = (args.client_id - 1) * chunk_size
    end_idx = start_idx + chunk_size if args.client_id != args.num_clients else len(texts)
    
    texts = texts[start_idx:end_idx]
    labels = labels[start_idx:end_idx]
    print(f"İstemci #{args.client_id} için {len(texts)} örnek ayrıldı (İndeks: {start_idx} - {end_idx}).")
    
    max_words = 5000
    max_length = 50
    x_padded, word_index, _ = tokenize_and_pad_texts(texts, max_words=max_words, max_length=max_length)
    
    split_idx = int(len(x_padded) * 0.8)
    x_train, y_train = x_padded[:split_idx], labels[:split_idx]
    x_test, y_test = x_padded[split_idx:], labels[split_idx:]

    vocab_size = min(max_words, len(word_index) + 1)
    
    # GloVe Vektörlerini Yükle
    embedding_matrix = None
    if os.path.exists(args.glove_path):
        embedding_matrix = load_glove_embeddings(args.glove_path, word_index, vocab_size, embedding_dim=100)
    else:
        print(f"UYARI: GloVe dosyası ({args.glove_path}) bulunamadı. Model kelimeleri sıfırdan öğrenecek.")

    model = build_bilstm_model(vocab_size=vocab_size, embedding_dim=100, max_length=max_length, embedding_matrix=embedding_matrix)

    client = PhishingClient(model, x_train, y_train, x_test, y_test, is_malicious=args.malicious)
    
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )

if __name__ == "__main__":
    main()
