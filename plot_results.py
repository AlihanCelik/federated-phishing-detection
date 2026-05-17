import json
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np


def main():
    if not os.path.exists("results.json"):
        print("HATA: 'results.json' bulunamadı. Önce simülasyonu çalıştırın.")
        sys.exit(1)

    with open("results.json", "r") as f:
        data = json.load(f)

    method        = data.get("method", "bilinmeyen")
    accuracies    = data.get("accuracies", [])
    losses        = data.get("losses", [])
    num_normal    = data.get("num_normal", "?")
    num_malicious = data.get("num_malicious", "?")
    num_rounds    = data.get("num_rounds", len(accuracies))

    # Label flipping istatistikleri
    lf = data.get("label_flipping", {})
    total_flipped      = lf.get("total_flipped_samples", "—")
    total_mal_rounds   = lf.get("total_malicious_rounds", "—")
    correctly_excluded = lf.get("correctly_excluded_rounds", "—")
    incorrectly_incl   = lf.get("incorrectly_included_rounds", "—")
    detection_rate     = lf.get("detection_rate_pct", None)

    if not accuracies:
        print("HATA: Doğruluk (accuracy) verisi bulunamadı.")
        sys.exit(1)

    rounds     = [item[0] for item in accuracies]
    acc_values = [item[1] * 100 for item in accuracies]

    max_acc   = max(acc_values)
    avg_acc   = np.mean(acc_values)
    final_acc = acc_values[-1]

    # Dalgalanma tespiti: ortalamadan >15 puan sapan roundları işaretle
    volatile_rounds = [
        rounds[i] for i, v in enumerate(acc_values)
        if abs(v - avg_acc) > 15
    ]

    # ------------------------------------------------------------------ #
    #  Layout: grafik üstte, tablo altta — GridSpec ile hizalı            #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[2.2, 1],   # grafik : tablo oranı
        hspace=0.08,
    )

    ax1 = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')          # tablo eksenini gizle

    # ------------------------------------------------------------------ #
    #  Grafik                                                              #
    # ------------------------------------------------------------------ #
    color_acc = 'tab:blue'
    ax1.set_xlabel('Round (Tur)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12)
    ax1.plot(rounds, acc_values, marker='o', linestyle='-',
             color=color_acc, linewidth=2, label='Accuracy', zorder=3)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, 115)

    # Ortalama accuracy yatay çizgisi
    ax1.axhline(y=avg_acc, color=color_acc, linestyle=':', alpha=0.5,
                label=f'Ortalama Accuracy ({avg_acc:.1f}%)')

    # Dalgalanan roundları kırmızı ile vurgula
    for r in volatile_rounds:
        ax1.axvspan(r - 0.4, r + 0.4, alpha=0.15, color='red')

    # Her noktanın üzerine değerini yaz
    for r, v in zip(rounds, acc_values):
        ax1.annotate(
            f"{v:.1f}%",
            (r, v),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=7,
            color=color_acc,
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
    ax1.set_title(
        f"Federatif Öğrenme Simülasyon Sonuçları\n"
        f"Savunma Yöntemi: {method_display}",
        fontsize=14, fontweight='bold', pad=12,
    )

    # Legend
    if volatile_rounds:
        volatile_patch = mpatches.Patch(
            color='red', alpha=0.3,
            label=f'Savunma zorlandı (Round: {volatile_rounds})'
        )
        ax1.legend(handles=[*ax1.get_lines(), volatile_patch],
                   loc='lower right', fontsize=9)
    else:
        ax1.legend(loc='lower right', fontsize=9)

    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    #  Tablo — ax_table eksenine yerleştir (kayma olmaz)                  #
    # ------------------------------------------------------------------ #
    if detection_rate is not None:
        det_str = f"%{detection_rate:.2f}"
        if detection_rate >= 80:
            det_color = '#1a7a1a'
        elif detection_rate >= 50:
            det_color = '#b87a00'
        else:
            det_color = '#b00000'
    else:
        det_str   = "—"
        det_color = 'black'

    table_data = [
        ["Savunma Yöntemi",                 method_display],
        ["Toplam Tur",                      str(num_rounds)],
        ["Normal İstemci",                  str(num_normal)],
        ["Kötü Niyetli İstemci",            str(num_malicious)],
        ["En Yüksek Accuracy",              f"%{max_acc:.2f}"],
        ["Ortalama Accuracy",               f"%{avg_acc:.2f}"],
        ["Son Tur Accuracy",                f"%{final_acc:.2f}"],
        ["── Label Flipping ──",            "──────────────────"],
        ["Üretilen Flip'li Örnek (toplam)", str(total_flipped)],
        ["Kötü Niyetli Katılım (round)",    str(total_mal_rounds)],
        ["Doğru Eleme (round)",             str(correctly_excluded)],
        ["Yanlış Seçim (round)",            str(incorrectly_incl)],
        ["Tespit Başarı Oranı",             det_str],
    ]

    col_labels  = ["Metrik", "Değer"]
    # Sütun genişlik oranları: sol sütun daha geniş
    col_widths  = [0.65, 0.35]

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        loc='center',           # ax_table'ın tam ortasına
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.35)          # satır yüksekliğini artır

    # Hücre renklendirme
    num_rows = len(table_data)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        if row == 0:
            # Başlık satırı
            cell.set_facecolor('#2c5f8a')
            cell.set_text_props(color='white', fontweight='bold')
        elif row == 8:
            # "── Label Flipping ──" ayraç satırı (0-indexed: başlık=0, veri 1'den)
            cell.set_facecolor('#d0e8ff')
            cell.set_text_props(color='#1a3a5c', fontweight='bold', style='italic')
        elif row == num_rows:
            # Son satır: Tespit Başarı Oranı
            cell.set_facecolor('#eaf4ea' if detection_rate and detection_rate >= 80
                               else '#fff4e0' if detection_rate and detection_rate >= 50
                               else '#fdecea')
            if col == 1:
                cell.set_text_props(color=det_color, fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f0f4f8')
        else:
            cell.set_facecolor('white')

    # ------------------------------------------------------------------ #
    #  Kaydet                                                              #
    # ------------------------------------------------------------------ #
    file_name = f"grafik_{method}.png"
    plt.savefig(file_name, dpi=150, bbox_inches='tight')
    print(f"[{method.upper()}] Grafik kaydedildi: {file_name}")
    print(f"  En yüksek accuracy : %{max_acc:.2f}")
    print(f"  Ortalama accuracy  : %{avg_acc:.2f}")
    print(f"  Son tur accuracy   : %{final_acc:.2f}")
    if volatile_rounds:
        print(f"  Savunma zorlandığı roundlar: {volatile_rounds}")
    if detection_rate is not None:
        print(f"\n  ── Label Flipping Tespit ──")
        print(f"  Üretilen flip'li örnek : {total_flipped}")
        print(f"  Doğru eleme oranı      : %{detection_rate:.2f} "
              f"({correctly_excluded}/{total_mal_rounds} round)")


if __name__ == "__main__":
    main()
