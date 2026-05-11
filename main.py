"""
Federatif Öğrenme Tabanlı Oltalama Tespiti
==========================================
Ana giriş noktası. Üç mod desteklenir:

  --prepare   : Global tokenizer oluşturur (simülasyondan önce bir kez çalıştırın)
  --simulate  : Federatif öğrenme simülasyonunu başlatır
  --plot      : Mevcut results.json dosyasından grafik oluşturur

Kullanım örnekleri:
  python main.py --prepare
  python main.py --simulate --num_normal 5 --num_malicious 1 --robust_method hybrid
  python main.py --plot
"""

import argparse
import subprocess
import sys
import os


def prepare(data_path: str, tokenizer_path: str = "tokenizer.pkl"):
    """
    Tüm veri seti üzerinde global tokenizer oluşturur ve diske kaydeder.
    Simülasyondan önce bir kez çalıştırılması yeterlidir.
    """
    from src.data_preprocessing import build_global_tokenizer

    if not os.path.exists(data_path):
        print(f"HATA: Veri seti bulunamadı: '{data_path}'")
        sys.exit(1)

    print(f"\n[Hazırlık] Global tokenizer oluşturuluyor: {data_path}")
    tokenizer = build_global_tokenizer(data_path, tokenizer_save_path=tokenizer_path)
    print(f"[Hazırlık] Tamamlandı. Kelime dağarcığı: {len(tokenizer.word_index)} kelime")
    print(f"[Hazırlık] Tokenizer kaydedildi: {tokenizer_path}")
    print("\nSimülasyonu başlatmak için:")
    print("  python main.py --simulate")


def simulate(args):
    """run_simulation.py'yi uygun argümanlarla çalıştırır."""
    cmd = [
        sys.executable, "run_simulation.py",
        "--num_normal", str(args.num_normal),
        "--num_malicious", str(args.num_malicious),
        "--robust_method", args.robust_method,
        "--num_rounds", str(args.num_rounds),
        "--alpha", str(args.alpha),
        "--data_path", args.data_path,
        "--glove_path", args.glove_path,
    ]
    subprocess.run(cmd)


def plot():
    """plot_results.py'yi çalıştırır."""
    if not os.path.exists("results.json"):
        print("HATA: 'results.json' bulunamadı. Önce simülasyonu çalıştırın.")
        sys.exit(1)
    subprocess.run([sys.executable, "plot_results.py"])


def main():
    parser = argparse.ArgumentParser(
        description="Federatif Öğrenme Tabanlı Oltalama Tespiti",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__,
    )

    # Mod seçimi
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--prepare", action="store_true",
                            help="Global tokenizer oluştur (simülasyondan önce bir kez çalıştır)")
    mode_group.add_argument("--simulate", action="store_true",
                            help="Federatif öğrenme simülasyonunu başlat")
    mode_group.add_argument("--plot", action="store_true",
                            help="Mevcut results.json'dan grafik oluştur")

    # Simülasyon parametreleri
    parser.add_argument("--num_normal", type=int, default=5,
                        help="Normal istemci sayısı (varsayılan: 5)")
    parser.add_argument("--num_malicious", type=int, default=1,
                        help="Kötü niyetli istemci sayısı (varsayılan: 1)")
    parser.add_argument("--robust_method", type=str, default="hybrid",
                        choices=["cosine", "krum", "hybrid"],
                        help="Savunma algoritması: cosine | krum | hybrid (varsayılan: hybrid)")
    parser.add_argument("--num_rounds", type=int, default=20,
                        help="Federatif öğrenme tur sayısı (varsayılan: 20)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet heterojenlik parametresi (varsayılan: 0.5)")
    parser.add_argument("--data_path", type=str, default="email_text.csv",
                        help="CSV veri seti yolu (varsayılan: email_text.csv)")
    parser.add_argument("--glove_path", type=str, default="glove.6B.100d.txt",
                        help="GloVe dosyası yolu (varsayılan: glove.6B.100d.txt)")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.pkl",
                        help="Tokenizer kayıt/yükleme yolu (varsayılan: tokenizer.pkl)")

    args = parser.parse_args()

    if args.prepare:
        prepare(args.data_path, args.tokenizer_path)
    elif args.simulate:
        simulate(args)
    elif args.plot:
        plot()


if __name__ == "__main__":
    main()
