import numpy as np
import os

def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    """
    GloVe dosyasını yükler ve veri setindeki kelimelere uygun 
    bir embedding matrisi oluşturur.
    """
    print(f"[{glove_path}] yükleniyor...")
    
    if not os.path.exists(glove_path):
        print(f"HATA: GloVe dosyası bulunamadı!")
        return None

    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f"Toplam {len(embeddings_index)} kelime vektörü GloVe dosyasından okundu.")

    # model için ağırlık matrisi
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    
    hits = 0
    misses = 0

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # kelime varsa matrise ekle
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            # kelime yoksa bırak
            misses += 1

    print(f"Eşleşen kelime sayısı: {hits} | Bulunamayan kelime sayısı: {misses}")
    return embedding_matrix