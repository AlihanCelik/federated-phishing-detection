import json
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def main():
    if not os.path.exists("results.json"):
        print("HATA: 'results.json' bulunamadı. Önce simülasyonu çalıştırın.")
        sys.exit(1)

    with open("results.json", "r") as f:
        data = json.load(f)

    method       = data.get("method", "bilinmeyen")
    accuracies   = data.get("accuracies", [])
    losses       = data.get("losses", [])
    num_normal   = data.get("num_normal", "?")
    num_malicious = data.get("num_malicious", "?")
    num_rounds   = data.get("num_rounds", len(accuracies))

    if not accuracies:
        print("HATA: Doğruluk (accuracy) verisi bulunamadı.")
        sys.exit(1)

    rounds     = [item[0] for item in accuracies]
    acc_values = [item[1] * 100 for item in accuracies]

    max_acc = max(acc_values)
    avg_acc = np.mean(acc_values)
    final_acc = acc_values[-1]

    # Dalgalanma tespiti: ortalamadan >15 puan sapan roundları işaretle
    volatile_rounds = [
        rounds[i] for i, v in enumerate(acc_values)
        if abs(v - avg_acc) > 15
    ]

    # ------------------------------------------------------------------ #
    #  Grafik                                                              #
    # ------------------------------------------------------------------ #
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Accuracy çizgisi
    color_acc = 'tab:blue'
    ax1.set_xlabel('Round (Tur)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12)
    ax1.plot(rounds, acc_values, marker='o', linestyle='-',
             color=color_acc, linewidth=2, label='Accuracy', zorder=3)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, 110)

    # Ortalama accuracy yatay çizgisi
    ax1.axhline(y=avg_acc, color=color_acc, linestyle=':', alpha=0.5,
                label=f'Ortalama Accuracy ({avg_acc:.1f}%)')

    # Dalgalanan roundları kırmızı ile vurgula
    for r in volatile_rounds:
        ax1.axvspan(r - 0.4, r + 0.4, alpha=0.15, color='red')

    # Her noktanın üzerine değerini yaz
    for i, (r, v) in enumerate(zip(rounds, acc_values)):
        ax1.annotate(
            f"{v:.1f}%",
            (r, v),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=7,
            color=color_acc
        )

    # Loss çizgisi (ikinci y ekseni)
    ax2 = ax1.twinx()
    color_loss = 'tab:red'
    if losses:
        loss_rounds = [item[0] for item in losses]
        loss_values = [item[1] for item in losses]
        ax2.set_ylabel('Loss', color=color_loss, fontsize=12)
        ax2.plot(loss_rounds, loss_values, marker='x', linestyle='--',
                 color=color_loss, linewidth=1.5, label='Loss', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color_loss)

    # Başlık
    method_labels = {
        "krum":   "Krum (Bizans Toleranslı)",
        "cosine": "Kosinüs Benzerliği",
        "hybrid": "Hibrit (Krum + Kosinüs)",
    }
    method_display = method_labels.get(method, method.upper())
    plt.title(
        f"Federatif Öğrenme Simülasyon Sonuçları\n"
        f"Savunma Yöntemi: {method_display}",
        fontsize=14, fontweight='bold', pad=15
    )

    # Açıklama kutusu
    if volatile_rounds:
        volatile_patch = mpatches.Patch(
            color='red', alpha=0.3,
            label=f'Savunma zorlandı (Round: {volatile_rounds})'
        )
        ax1.legend(handles=[
            *ax1.get_lines(),
            volatile_patch
        ], loc='lower right', fontsize=9)
    else:
        ax1.legend(loc='lower right', fontsize=9)

    # Özet tablo
    table_data = [
        ["Savunma Yöntemi",        method_display],
        ["Toplam Tur",             str(num_rounds)],
        ["Normal İstemci",         str(num_normal)],
        ["Kötü Niyetli İstemci",   str(num_malicious)],
        ["En Yüksek Accuracy",     f"%{max_acc:.2f}"],
        ["Ortalama Accuracy",      f"%{avg_acc:.2f}"],
        ["Son Tur Accuracy",       f"%{final_acc:.2f}"],
    ]

    table = plt.table(
        cellText=table_data,
        colLabels=["Metrik", "Değer"],
        loc='bottom',
        bbox=[0.1, -0.52, 0.8, 0.35],
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c5f8a')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f0f4f8')

    fig.subplots_adjust(bottom=0.42)
    plt.grid(True, alpha=0.3)

    file_name = f"grafik_{method}.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"[{method.upper()}] Grafik kaydedildi: {file_name}")
    print(f"  En yüksek accuracy : %{max_acc:.2f}")
    print(f"  Ortalama accuracy  : %{avg_acc:.2f}")
    print(f"  Son tur accuracy   : %{final_acc:.2f}")
    if volatile_rounds:
        print(f"  Savunma zorlandığı roundlar: {volatile_rounds}")


if __name__ == "__main__":
    main()
