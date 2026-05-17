import subprocess
import time
import sys
import argparse
import os
import socket


def wait_for_server(host: str = "127.0.0.1", port: int = 8080,
                    timeout: int = 120, interval: float = 2.0):
    """
    Sunucunun belirtilen port'u dinlemeye başlamasını bekler.
    Her `interval` saniyede bir bağlantı dener, `timeout` saniye sonra vazgeçer.
    """
    print(f"[Bekliyor] Sunucu {host}:{port} adresinde hazır olana kadar bekleniyor...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"[Hazır] Sunucu {host}:{port} bağlantıya hazır.", flush=True)
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(interval)
    print(f"[HATA] Sunucu {timeout} saniye içinde başlamadı!", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Federatif Oltalama Tespiti Simülasyon Başlatıcı",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--num_normal", type=int, default=5,
                        help="Normal (güvenilir) istemci sayısı")
    parser.add_argument("--num_malicious", type=int, default=1,
                        help="Kötü niyetli istemci sayısı (label flipping saldırısı)")
    parser.add_argument("--robust_method", type=str, default="fltrust",
                        choices=["cosine", "krum", "hybrid", "fltrust"],
                        help="Savunma algoritması: cosine | krum | hybrid | fltrust")
    parser.add_argument("--num_rounds", type=int, default=50,
                        help="Federatif öğrenme tur sayısı")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet heterojenlik parametresi (küçük=heterojen)")
    parser.add_argument("--data_path", type=str, default="email_text.csv",
                        help="CSV veri seti yolu")
    parser.add_argument("--glove_path", type=str, default="glove.6B.100d.txt",
                        help="GloVe dosyası yolu")
    parser.add_argument("--server_data_size", type=int, default=100,
                        help="FLTrust için sunucu temiz veri seti boyutu (varsayılan: 100)")
    args = parser.parse_args()

    total_clients = args.num_normal + args.num_malicious
    malicious_fraction = args.num_malicious / total_clients if total_clients > 0 else 0.0

    # Krum güvenlik koşulunu önceden kontrol et
    if args.robust_method in ("krum", "hybrid"):
        threshold = (total_clients - 2) / 2
        if args.num_malicious >= threshold:
            print(
                f"[UYARI] Krum güvenlik koşulu: f={args.num_malicious} >= (n-2)/2={threshold:.1f}\n"
                f"        Savunma zayıflayabilir. Daha fazla normal istemci eklemeyi düşünün."
            )

    print(f"\n{'='*60}")
    print(f"  Federatif Oltalama Tespiti Simülasyonu Başlatılıyor")
    print(f"{'='*60}")
    print(f"  Toplam İstemci  : {total_clients} "
          f"(Normal: {args.num_normal}, Kötü Niyetli: {args.num_malicious})")
    print(f"  Savunma Yöntemi : {args.robust_method.upper()}")
    print(f"  Tur Sayısı      : {args.num_rounds}")
    print(f"  Heterojenlik (α): {args.alpha}")
    print(f"{'='*60}\n")

    # --- Global tokenizer oluştur (yoksa) ---
    tokenizer_path = "tokenizer.pkl"
    if not os.path.exists(tokenizer_path):
        print("[Ön İşlem] Global tokenizer oluşturuluyor...")
        result = subprocess.run(
            [sys.executable, "-c",
             f"from src.data_preprocessing import build_global_tokenizer; "
             f"build_global_tokenizer('{args.data_path}')"]
        )
        if result.returncode != 0:
            print("HATA: Tokenizer oluşturulamadı. Simülasyon durduruluyor.")
            sys.exit(1)
    else:
        print(f"[Ön İşlem] Mevcut tokenizer kullanılıyor: {tokenizer_path}")

    # --- Sunucuyu başlat ---
    print("\n[1] Sunucu (server.py) başlatılıyor...")
    server_cmd = [
        sys.executable, "server.py",
        "--num_clients", str(total_clients),
        "--malicious_fraction", str(malicious_fraction),
        "--robust_method", args.robust_method,
        "--num_rounds", str(args.num_rounds),
        "--data_path", args.data_path,
        "--glove_path", args.glove_path,
        "--server_data_size", str(args.server_data_size),
    ]
    server_process = subprocess.Popen(server_cmd)

    # Sunucunun port'u dinlemeye başlamasını bekle (sabit sleep yerine aktif kontrol)
    if not wait_for_server(timeout=180):
        print("Sunucu başlatılamadı. Simülasyon durduruluyor.")
        server_process.terminate()
        sys.exit(1)

    clients = []

    # --- Normal istemcileri başlat ---
    for i in range(1, args.num_normal + 1):
        print(f"[{i + 1}] Normal İstemci #{i} başlatılıyor...")
        proc = subprocess.Popen([
            sys.executable, "client.py",
            "--client_id", str(i),
            "--num_clients", str(total_clients),
            "--data_path", args.data_path,
            "--glove_path", args.glove_path,
            "--tokenizer_path", tokenizer_path,
            "--alpha", str(args.alpha),
        ])
        clients.append(proc)
        time.sleep(2)

    # --- Kötü niyetli istemcileri başlat ---
    for i in range(1, args.num_malicious + 1):
        client_id = args.num_normal + i
        print(f"[{client_id + 1}] Kötü Niyetli İstemci #{client_id} başlatılıyor "
              f"(Label Flipping saldırısı)...")
        proc = subprocess.Popen([
            sys.executable, "client.py",
            "--client_id", str(client_id),
            "--malicious",
            "--num_clients", str(total_clients),
            "--data_path", args.data_path,
            "--glove_path", args.glove_path,
            "--tokenizer_path", tokenizer_path,
            "--alpha", str(args.alpha),
        ])
        clients.append(proc)
        time.sleep(2)

    # --- Tüm istemcilerin bitmesini bekle ---
    try:
        for client in clients:
            client.wait()

        server_process.terminate()
        server_process.wait()

        print(f"\n{'='*60}")
        print("  Simülasyon Tamamlandı")
        print(f"{'='*60}")

        print("\nSonuç grafiği oluşturuluyor...")
        subprocess.run([sys.executable, "plot_results.py"])

    except KeyboardInterrupt:
        print("\nKullanıcı tarafından iptal edildi. Tüm süreçler kapatılıyor...")
        for client in clients:
            client.terminate()
        server_process.terminate()


if __name__ == "__main__":
    main()
