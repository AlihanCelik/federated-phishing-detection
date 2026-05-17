import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout


def build_bilstm_model(vocab_size: int, embedding_dim: int,
                       max_length: int, embedding_matrix=None) -> Sequential:
    """
    GloVe destekli BiLSTM modeli oluşturur.

    Mimari:
        Embedding (GloVe, trainable=False)
          → BiLSTM (64 hücre: 32 ileri + 32 geri)
          → Dropout (0.5)
          → Dense (1, sigmoid) — ikili sınıflandırma

    Neden BiLSTM?
        Oltalama e-postalarındaki manipülatif dil cümlenin geneline yayılır.
        BiLSTM metni hem soldan sağa hem sağdan sola işleyerek uzak kelimeler
        arasındaki anlamsal köprüleri kurar. GloVe vektörleri ise kelimeler
        arası matematiksel yakınlığı önceden öğrenilmiş biçimde sağlar.

    Args:
        vocab_size:       Kelime dağarcığı boyutu
        embedding_dim:    GloVe vektör boyutu (100)
        max_length:       Maksimum dizi uzunluğu
        embedding_matrix: Önceden eğitilmiş GloVe ağırlık matrisi (opsiyonel)

    Returns:
        Derlenmiş Keras Sequential modeli
    """
    model = Sequential(name="BiLSTM_Phishing_Detector")

    # Embedding katmanı
    if embedding_matrix is not None:
        # GloVe ağırlıkları yüklendi; embedding katmanı donduruldu (trainable=False).
        # Bu, GloVe'un önceden öğrendiği anlamsal ilişkilerin bozulmasını engeller.
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=False,
            name="glove_embedding"
        ))
    else:
        # GloVe yoksa embedding sıfırdan öğrenilir
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            name="random_embedding"
        ))

    # BiLSTM katmanı: 64 hücre (32 ileri + 32 geri)
    model.add(Bidirectional(LSTM(64, return_sequences=False), name="bilstm"))

    # Overfitting'i önlemek için Dropout
    model.add(Dropout(0.5, name="dropout"))

    # Çıkış katmanı: sigmoid → oltalama olasılığı (0-1)
    model.add(Dense(1, activation='sigmoid', name="output"))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


# ---------------------------------------------------------------------------
# NOT: tokenize_and_pad_texts fonksiyonu bu modülden kaldırıldı.
# Tüm tokenizasyon işlemleri artık src/data_preprocessing.py içindeki
# global tokenizer üzerinden yapılmaktadır. Bu sayede tüm istemciler
# aynı kelime-indeks eşleşmesini kullanır ve model ağırlıkları tutarlı kalır.
# ---------------------------------------------------------------------------
