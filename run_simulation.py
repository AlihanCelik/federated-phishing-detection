import subprocess
import time
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Simülasyon Başlatıcı")
    parser.add_argument("--num_normal", type=int, default=2, help="Normal istemci sayısı")
    parser.add_argument("--num_malicious", type=int, default=1, help="Kötü niyetli istemci sayısı")
    parser.add_argument("--robust_method", type=str, default="cosine", choices=["cosine", "krum"], help="Kullanılacak savunma algoritması (cosine veya krum)")
    args = parser.parse_args()

    total_clients = args.num_normal + args.num_malicious
    malicious_fraction = args.num_malicious / total_clients if total_clients > 0 else 0.0

    print(f"=== Federatif Oltalama Tespiti Simülasyonu Başlatılıyor ===")
    print(f"Toplam İstemci: {total_clients} (Normal: {args.num_normal}, Kötü Niyetli: {args.num_malicious})")
    print(f"Kullanılacak Savunma Algoritması: {args.robust_method.upper()}")
    
    print("[1] Sunucu (server.py) başlatılıyor...")
    server_process = subprocess.Popen([
        sys.executable, "server.py", 
        "--num_clients", str(total_clients), 
        "--malicious_fraction", str(malicious_fraction),
        "--robust_method", args.robust_method
    ])

    
    time.sleep(5)
    
    clients = []
    
    # Normal İstemciler
    for i in range(1, args.num_normal + 1):
        print(f"[{i+1}] Normal İstemci #{i} başlatılıyor...")
        client_process = subprocess.Popen([
            sys.executable, "client.py", 
            "--client_id", str(i), 
            "--num_clients", str(total_clients), 
            "--data_path", "email_text.csv"
        ])
        clients.append(client_process)
        time.sleep(2)
        
    # Kötü Niyetli İstemciler
    for i in range(1, args.num_malicious + 1):
        client_id = args.num_normal + i
        print(f"[{client_id+1}] Kötü Niyetli İstemci #{client_id} başlatılıyor... (Zehirleme saldırısı)")
        malicious_client = subprocess.Popen([
            sys.executable, "client.py", 
            "--client_id", str(client_id), 
            "--malicious", 
            "--num_clients", str(total_clients), 
            "--data_path", "email_text.csv"
        ])
        clients.append(malicious_client)
        time.sleep(2)

    try:
        for client in clients:
            client.wait()
            
        server_process.terminate()
        server_process.wait()
        print("=== Simülasyon Tamamlandı ===")
        
    except KeyboardInterrupt:
        print("Kullanıcı tarafından iptal edildi. Süreçler kapatılıyor...")
        for client in clients:
            client.terminate()
        server_process.terminate()

if __name__ == "__main__":
    main()
