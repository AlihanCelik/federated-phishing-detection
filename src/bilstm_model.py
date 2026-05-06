import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def build_bilstm_model(vocab_size, embedding_dim, max_length, embedding_matrix=None):
    
    # kelime dağarcığı boyutu,embedding vektörlerinin boyutu,maksimum dizi uzunluğu,önceden eğitilmiş GloVe ağırlık matrisi.
  
    model = Sequential()
    
    # Embedding Katmanı
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)) 
    else:
        model.add(Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=max_length))
    
    # BiLSTM Katmanı 64 hücre (32 ileri, 32 geri)
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    
    #overfitting engellemek için Dropout
    model.add(Dropout(0.5))
    
    # binary classification: oltalama veya temiz
    model.add(Dense(1, activation='sigmoid'))
    
    # model derleme
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def tokenize_and_pad_texts(texts, max_words=10000, max_length=100):

    # metinleri tokenleştirir ve padler
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    return padded_sequences, tokenizer.word_index, tokenizer
