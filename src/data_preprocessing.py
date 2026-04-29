import re
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def clean_text(text):
    """
    URL ve başlık bilgilerini dışlar
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    
    # URL ve linkleri kaldırma 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # HTML etiketlerini ve e-posta adreslerini temizl
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # sadece alfabetik karakterleri tut
    text = re.sub(r'[^a-z\s]', '', text)

    # stopwords temizliği
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)


def process_and_save_dataset(input_csv, output_csv, text_column, label_column):
    """
    tüm veri setini temizler ve yeni bir dosya olarak kaydeder

    """
    print(f"[{input_csv}] dosyası yükleniyor...")
    
    if not os.path.exists(input_csv):
        print(f"HATA: '{input_csv}' bulunamadı! Lütfen veri setini 'data' klasörüne ekleyin.")
        return

    df = pd.read_csv(input_csv)
    
    if text_column not in df.columns or label_column not in df.columns:
        print(f"HATA: Veri setinde '{text_column}' veya '{label_column}' sütunu bulunamadı.")
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
    INPUT_PATH = "data/email_text.csv"       
    OUTPUT_PATH = "data/processed_dataset.csv"     
    TEXT_COL = "text"   
    LABEL_COL = "label" 

    process_and_save_dataset(INPUT_PATH, OUTPUT_PATH, TEXT_COL, LABEL_COL)