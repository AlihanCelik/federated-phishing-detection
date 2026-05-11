import json
import matplotlib.pyplot as plt
import os
import sys

def main():
    if not os.path.exists("results.json"):
        print("Sonuç dosyası bulunamadı (results.json). Önce simülasyonu çalıştırın.")
        sys.exit(1)

    with open("results.json", "r") as f:
        data = json.load(f)

    method = data.get("method", "Bilinmeyen")
    accuracies = data.get("accuracies", [])
    losses = data.get("losses", [])

    if not accuracies:
        print("Doğruluk (accuracy) verisi bulunamadı.")
        sys.exit(1)

    num_normal = data.get("num_normal", "?")
    num_malicious = data.get("num_malicious", "?")
    num_rounds = len(accuracies)
    
    rounds = [item[0] for item in accuracies]
    acc_values = [item[1] * 100 for item in accuracies]  # Yüzdeye çevir
    
    max_accuracy = f"{max(acc_values):.2f}%" if acc_values else "?"

    # Çift y eksenli (accuracy ve loss) grafik oluştur
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Round (Tur)')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(rounds, acc_values, marker='o', linestyle='-', color=color, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Her noktanın üzerine değerini yaz
    for i, txt in enumerate(acc_values):
        ax1.annotate(f"{txt:.1f}%", (rounds[i], acc_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

    ax2 = ax1.twinx()  
    color = 'tab:red'
    
    if losses:
        loss_rounds = [item[0] for item in losses]
        loss_values = [item[1] for item in losses]
        ax2.set_ylabel('Loss', color=color)  
        ax2.plot(loss_rounds, loss_values, marker='x', linestyle='--', color=color, linewidth=2, label='Loss')
        ax2.tick_params(axis='y', labelcolor=color)

    # Başlık ve tablo
    plt.title(f"Federatif Öğrenme Simülasyon Sonuçları\nKullanılan Algoritma: {method.upper()}", fontsize=14, fontweight='bold')
    
    # Tablo verilerini hazırla
    table_data = [
        ["Kullanılan Algoritma", method.upper()],
        ["Toplam Tur (Round)", num_rounds],
        ["Normal İstemci Sayısı", num_normal],
        ["Zehirli İstemci Sayısı", num_malicious],
        ["Ulaşılan En Yüksek Başarı", max_accuracy]
    ]

    # Tabloyu grafiğin altına çiz
    table = plt.table(cellText=table_data, colLabels=["Metrik", "Değer"], loc='bottom', bbox=[0.2, -0.45, 0.6, 0.3], cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    # Alt boşluğu artır ki tablo grafikle çakışmasın
    fig.subplots_adjust(bottom=0.35)
    
    plt.grid(True, alpha=0.3)
    
    # Grafiği resim olarak kaydet (bekleme yapmaması için plt.show() kaldırıldı)
    file_name = f"grafik_{method}.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"[{method.upper()}] Sonuç grafiği başarıyla kaydedildi: {file_name}")

if __name__ == "__main__":
    main()
