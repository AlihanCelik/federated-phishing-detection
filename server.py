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
            
        else:
            print(f"Bilinmeyen yöntem: {self.robust_method}. Standart FedAvg uygulanıyor.")
            return super().aggregate_fit(server_round, results, failures)

        # sadece seçilen istemcilerin sonuçlarını filtrele
        robust_results = [results[i] for i in selected_indices]
        
        # seçilen istemcilerle FedAvg metodunu çağırarak ortalama al
        return super().aggregate_fit(server_round, robust_results, failures)

def main():
    print("Federatif Öğrenme Sunucusu Başlatılıyor...")
    strategy = CustomRobustStrategy(
        robust_method="cosine", 
        malicious_fraction=0.3,
        min_fit_clients=3,
        min_available_clients=3,
        min_evaluate_clients=3,
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    import typing
    from typing import Union
    main()
