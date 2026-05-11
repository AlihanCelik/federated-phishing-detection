import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

class CustomRobustStrategy(FedAvg):
    def __init__(self, robust_method="cosine", malicious_fraction=0.3, **kwargs):
      
        #bizans toleranslı savunma katmanı
    
        super().__init__(**kwargs)
        self.robust_method = robust_method
        self.malicious_fraction = malicious_fraction

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # gelen ağırlıkları numpy array'e çeviriyoruz
        client_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        
        # katman ağırlıklarını tek bir 1B vektörde birleştirme fonksiyonu
        def flatten_weights(weights):
            return np.concatenate([w.flatten() for w in weights])

        flat_weights = np.array([flatten_weights(w) for w in client_weights])
        num_clients = len(flat_weights)
        import math
        num_malicious = math.ceil(num_clients * self.malicious_fraction)
        
        # eğer yeterli istemci yoksa filtreleme yapmadan standart ortalama al 
        if num_clients < 3 or num_malicious == 0:
            print(f"Yuva {server_round}: Yeterli istemci ({num_clients}) veya kötü niyetli oranı yok, standart FedAvg uygulanıyor.")
            return super().aggregate_fit(server_round, results, failures)

        selected_indices = []

        if self.robust_method == "krum":
            print(f"\n--- Yuva {server_round}: Krum Algoritması Uygulanıyor ---")
            # her istemci için diğer istemcilere olan Öklid uzaklıklarını hesapla
            distances = np.zeros((num_clients, num_clients))
            for i in range(num_clients):
                for j in range(num_clients):
                    if i != j:
                        distances[i, j] = np.linalg.norm(flat_weights[i] - flat_weights[j])
            
            # krum: Her düğüm için en yakın düğümün mesafesini topla.
            # multi-krum için en düşük puana sahip k düğümü seçeceğiz.
            krum_scores = np.zeros(num_clients)
            neighbors_to_keep = max(1, num_clients - num_malicious - 2)
            
            for i in range(num_clients):
                sorted_dists = np.sort(distances[i])
                # en yakın komşuların uzaklık toplamı 
                krum_scores[i] = np.sum(sorted_dists[1:neighbors_to_keep+1])
            
            # en düşük skora sahip olan güvenilir düğümleri seç 
            # kaç düğüm seçeceğimiz: num_clients - num_malicious
            num_to_select = max(1, num_clients - num_malicious)
            selected_indices = np.argsort(krum_scores)[:num_to_select].tolist()
            print(f"Krum Skorları: {krum_scores}")
            print(f"Seçilen İstemci İndeksleri: {selected_indices}")

        elif self.robust_method == "cosine":
            print(f"\n---Round {server_round}: Kosinüs Benzerliği Uygulanıyor ---")
            # referans vektör olarak tüm güncellemelerin ortalamasını al
            mean_vector = np.mean(flat_weights, axis=0)
            
            similarities = np.zeros(num_clients)
            for i in range(num_clients):
                dot_product = np.dot(flat_weights[i], mean_vector)
                norm_a = np.linalg.norm(flat_weights[i])
                norm_b = np.linalg.norm(mean_vector)
                
                # kosinüs benzerliği hesapla
                if norm_a == 0 or norm_b == 0:
                    similarities[i] = 0
                else:
                    similarities[i] = dot_product / (norm_a * norm_b)
            
            print(f"Kosinüs Benzerlikleri: {similarities}")
            # en yüksek benzerliğe sahip olan (num_clients - num_malicious) düğümü seç
            num_to_select = max(1, num_clients - num_malicious)
            selected_indices = np.argsort(similarities)[-num_to_select:].tolist()
            print(f"Seçilen İstemci İndeksleri: {selected_indices}")
            
        elif self.robust_method == "hybrid":
            print(f"\n---Round {server_round}: Hibrit (Krum + Kosinüs) Savunma Uygulanıyor ---")
            
            # Krum hesaplama
            distances = np.zeros((num_clients, num_clients))
            for i in range(num_clients):
                for j in range(num_clients):
                    if i != j:
                        distances[i, j] = np.linalg.norm(flat_weights[i] - flat_weights[j])
            
            krum_scores = np.zeros(num_clients)
            neighbors_to_keep = max(1, num_clients - num_malicious - 2)
            for i in range(num_clients):
                sorted_dists = np.sort(distances[i])
                krum_scores[i] = np.sum(sorted_dists[1:neighbors_to_keep+1])
            
            # Krum skorlarını normalize et (Düşük mesafe = yüksek puan, 0-1 arası)
            min_krum, max_krum = np.min(krum_scores), np.max(krum_scores)
            if max_krum == min_krum:
                krum_norm = np.ones(num_clients)
            else:
                krum_norm = 1.0 - ((krum_scores - min_krum) / (max_krum - min_krum))
            
            # Kosinüs hesaplama
            mean_vector = np.mean(flat_weights, axis=0)
            similarities = np.zeros(num_clients)
            for i in range(num_clients):
                dot_product = np.dot(flat_weights[i], mean_vector)
                norm_a = np.linalg.norm(flat_weights[i])
                norm_b = np.linalg.norm(mean_vector)
                if norm_a == 0 or norm_b == 0:
                    similarities[i] = 0
                else:
                    similarities[i] = dot_product / (norm_a * norm_b)
            
            # Kosinüs skorlarını normalize et (0-1 arası)
            min_cos, max_cos = np.min(similarities), np.max(similarities)
            if max_cos == min_cos:
                cos_norm = np.ones(num_clients)
            else:
                cos_norm = (similarities - min_cos) / (max_cos - min_cos)
            
            # Hibrit Skor = %50 Krum + %50 Kosinüs
            hybrid_scores = (0.5 * krum_norm) + (0.5 * cos_norm)
            
            print(f"Krum Puanları: {krum_norm}")
            print(f"Kosinüs Puanları: {cos_norm}")
            print(f"Hibrit Skorları: {hybrid_scores}")
            
            num_to_select = max(1, num_clients - num_malicious)
            selected_indices = np.argsort(hybrid_scores)[-num_to_select:].tolist()
            print(f"Seçilen İstemci İndeksleri: {selected_indices}")
            
        else:
            print(f"Bilinmeyen yöntem: {self.robust_method}. Standart FedAvg uygulanıyor.")
            return super().aggregate_fit(server_round, results, failures)

        # sadece seçilen istemcilerin sonuçlarını filtrele
        robust_results = [results[i] for i in selected_indices]
        
        # seçilen istemcilerle FedAvg metodunu çağırarak ortalama al
        return super().aggregate_fit(server_round, robust_results, failures)

def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy = sum(accuracies) / sum(examples)
    print(f"\n--- [SUNUCU] Global Model Başarı Oranı (Accuracy): %{accuracy * 100:.2f} ---\n")
    return {"accuracy": accuracy}

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Federatif Öğrenme Sunucusu")
    parser.add_argument("--num_clients", type=int, default=3, help="Toplam istemci sayısı")
    parser.add_argument("--malicious_fraction", type=float, default=0.3, help="Kötü niyetli istemci oranı")
    parser.add_argument("--robust_method", type=str, default="cosine", choices=["cosine", "krum", "hybrid"], help="Kullanılacak savunma algoritması (cosine, krum veya hybrid)")
    args = parser.parse_args()

    print(f"Federatif Öğrenme Sunucusu Başlatılıyor... (Beklenen İstemci: {args.num_clients}, Algoritma: {args.robust_method})")
    strategy = CustomRobustStrategy(
        robust_method=args.robust_method, 
        malicious_fraction=args.malicious_fraction,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average
    )
    
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )
    import math
    num_malicious = math.ceil(args.num_clients * args.malicious_fraction)
    num_normal = args.num_clients - num_malicious

    # Sonuçları kaydet
    results_data = {
        "method": args.robust_method,
        "num_clients": args.num_clients,
        "num_normal": num_normal,
        "num_malicious": num_malicious,
        "losses": history.losses_distributed,
        "accuracies": history.metrics_distributed.get("accuracy", [])
    }
    with open("results.json", "w") as f:
        json.dump(results_data, f)

if __name__ == "__main__":
    import typing
    from typing import Union
    main()
