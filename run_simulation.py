import subprocess
import time
import sys

def main():
    print("=== Federatif Oltalama Tespiti Simülasyonu Başlatılıyor ===")
    
    print("[1] Sunucu (server.py) başlatılıyor...")
    server_process = subprocess.Popen([sys.executable, "server.py"])
    
    time.sleep(5)
    
    clients = []
    
    # 2 Adet Normal İstemci
    for i in range(1, 3):
        print("[{i+1}] Normal İstemci #{i} başlatılıyor...")
        client_process = subprocess.Popen([sys.executable, "client.py", "--client_id", str(i), "--num_clients", "3", "--data_path", "email_text.csv"])
        clients.append(client_process)
        time.sleep(2)
        
    # 1 Adet Kötü Niyetli İstemci
    print(f"Kötü Niyetli İstemci #3 başlatılıyor... (Zehirleme saldırısı simülasyonu)")
    malicious_client = subprocess.Popen([sys.executable, "client.py", "--client_id", "3", "--malicious", "--num_clients", "3", "--data_path", "email_text.csv"])
    clients.append(malicious_client)

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
