import math
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy


def check_krum_feasibility(num_clients: int, num_malicious: int) -> bool:
    """
    Krum algoritmasının güvenlik garantisi için gerekli koşulu kontrol eder.

    Krum teorik güvencesi: f < (n - 2) / 2
    Yani kötü niyetli istemci sayısı (f), toplam istemci sayısının (n)
    yarısından 1 eksiğinden az olmalıdır.

    Referans: Blanchard et al., "Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent", NeurIPS 2017.
    """
    threshold = (num_clients - 2) / 2
    feasible = num_malicious < threshold
    if not feasible:
        print(
            f"[UYARI] Krum güvenlik koşulu sağlanamıyor: "
            f"f={num_malicious} >= (n-2)/2={threshold:.1f}. "
            f"Kötü niyetli istemci oranı çok yüksek, savunma zayıflayabilir."
        )
    else:
        print(
            f"[Krum] Güvenlik koşulu sağlandı: "
            f"f={num_malicious} < (n-2)/2={threshold:.1f}"
        )
    return feasible


class CustomRobustStrategy(FedAvg):
    """
    Bizans Toleranslı Federatif Öğrenme Stratejisi.

    Standart FedAvg'ın üzerine inşa edilmiş üç savunma yöntemi sunar:
    - krum:   Öklid uzaklığına dayalı anomali tespiti (Blanchard et al., 2017)
    - cosine: Kosinüs benzerliğine dayalı yön anomalisi tespiti
    - hybrid: Krum + Kosinüs skorlarının ağırlıklı ortalaması (%50/%50)

    Her yöntem, çoğunluktan sapan (zehirli) gradyanları dışlayarak
    yalnızca güvenilir istemcilerin güncellemelerini birleştirir.
    """

    def __init__(self, robust_method: str = "hybrid",
                 malicious_fraction: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.robust_method = robust_method
        self.malicious_fraction = malicious_fraction

        # ---- Label Flipping Tespit İstatistikleri ----
        # Her round'da istemcilerden gelen metadata toplanır.
        # Simülasyonda istemci kendi is_malicious bilgisini gönderir;
        # bu sayede savunmanın gerçek kötü niyetlileri kaç roundda
        # doğru elediği ölçülebilir.
        self.detection_stats = {
            # flipped_count istemci başlatılırken bir kez hesaplanır.
            # Sunucu ilk round'da bu değeri alır ve sabit tutar — her round
            # tekrar toplamaz. Böylece "20 round × 279 flip = 5580" hatası olmaz.
            "total_flipped_samples":       0,   # kötü niyetli istemci başına flip sayısı (sabit)
            "flipped_samples_recorded":    False,# ilk round'da bir kez kaydedildi mi
            "total_malicious_rounds":      0,   # kötü niyetli istemcinin katıldığı round sayısı
            "correctly_excluded_rounds":   0,   # kötü niyetlinin doğru elendi round sayısı
            "incorrectly_included_rounds": 0,   # kötü niyetlinin yanlışlıkla seçildi round sayısı
            "per_round": [],                    # round bazlı detay
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Gelen ağırlıkları numpy array'e çevir
        client_weights = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]

        def flatten_weights(weights):
            """Katman ağırlıklarını tek bir 1B vektörde birleştirir."""
            return np.concatenate([w.flatten() for w in weights])

        flat_weights = np.array([flatten_weights(w) for w in client_weights])
        num_clients = len(flat_weights)
        num_malicious = math.ceil(num_clients * self.malicious_fraction)

        # Yeterli istemci yoksa standart FedAvg uygula
        if num_clients < 3 or num_malicious == 0:
            print(f"[Round {server_round}] Yeterli istemci yok, standart FedAvg uygulanıyor.")
            return super().aggregate_fit(server_round, results, failures)

        selected_indices = []

        # ------------------------------------------------------------------ #
        #  KRUM                                                                #
        # ------------------------------------------------------------------ #
        if self.robust_method == "krum":
            print(f"\n--- [Round {server_round}] Krum Algoritması Uygulanıyor ---")
            check_krum_feasibility(num_clients, num_malicious)

            # Her istemci çifti arasındaki Öklid uzaklığını hesapla
            distances = np.zeros((num_clients, num_clients))
            for i in range(num_clients):
                for j in range(num_clients):
                    if i != j:
                        distances[i, j] = np.linalg.norm(
                            flat_weights[i] - flat_weights[j]
                        )

            # Krum skoru: her düğüm için en yakın (n - f - 2) komşunun uzaklık toplamı
            neighbors_to_keep = max(1, num_clients - num_malicious - 2)
            krum_scores = np.zeros(num_clients)
            for i in range(num_clients):
                sorted_dists = np.sort(distances[i])
                krum_scores[i] = np.sum(sorted_dists[1: neighbors_to_keep + 1])

            # En düşük skora sahip (num_clients - num_malicious) istemciyi seç
            num_to_select = max(1, num_clients - num_malicious)
            selected_indices = np.argsort(krum_scores)[:num_to_select].tolist()

            print(f"  Krum Skorları:            {np.round(krum_scores, 4)}")
            print(f"  Seçilen İstemci İndeksleri: {selected_indices}")

        # ------------------------------------------------------------------ #
        #  COSİNÜS BENZERLİĞİ                                                 #
        # ------------------------------------------------------------------ #
        elif self.robust_method == "cosine":
            print(f"\n--- [Round {server_round}] Kosinüs Benzerliği Uygulanıyor ---")

            # Referans vektör: tüm güncellemelerin ortalaması
            mean_vector = np.mean(flat_weights, axis=0)

            similarities = np.zeros(num_clients)
            for i in range(num_clients):
                norm_a = np.linalg.norm(flat_weights[i])
                norm_b = np.linalg.norm(mean_vector)
                if norm_a == 0 or norm_b == 0:
                    similarities[i] = 0.0
                else:
                    similarities[i] = (
                        np.dot(flat_weights[i], mean_vector) / (norm_a * norm_b)
                    )

            # En yüksek benzerliğe sahip (num_clients - num_malicious) istemciyi seç
            num_to_select = max(1, num_clients - num_malicious)
            selected_indices = np.argsort(similarities)[-num_to_select:].tolist()

            print(f"  Kosinüs Benzerlikleri:      {np.round(similarities, 4)}")
            print(f"  Seçilen İstemci İndeksleri: {selected_indices}")

        # ------------------------------------------------------------------ #
        #  HİBRİT (KRUM + COSİNÜS)                                            #
        # ------------------------------------------------------------------ #
        elif self.robust_method == "hybrid":
            print(f"\n--- [Round {server_round}] Hibrit Savunma (Krum + Kosinüs) Uygulanıyor ---")
            check_krum_feasibility(num_clients, num_malicious)

            # -- Krum hesaplama --
            distances = np.zeros((num_clients, num_clients))
            for i in range(num_clients):
                for j in range(num_clients):
                    if i != j:
                        distances[i, j] = np.linalg.norm(
                            flat_weights[i] - flat_weights[j]
                        )

            neighbors_to_keep = max(1, num_clients - num_malicious - 2)
            krum_scores = np.zeros(num_clients)
            for i in range(num_clients):
                sorted_dists = np.sort(distances[i])
                krum_scores[i] = np.sum(sorted_dists[1: neighbors_to_keep + 1])

            # Krum skorlarını normalize et: düşük mesafe → yüksek puan (0-1)
            min_k, max_k = np.min(krum_scores), np.max(krum_scores)
            krum_norm = (
                np.ones(num_clients)
                if max_k == min_k
                else 1.0 - (krum_scores - min_k) / (max_k - min_k)
            )

            # -- Kosinüs hesaplama --
            # Medyan vektör: ortalamaya göre aykırı değerlere daha dayanıklı
            median_vector = np.median(flat_weights, axis=0)
            similarities = np.zeros(num_clients)
            for i in range(num_clients):
                norm_a = np.linalg.norm(flat_weights[i])
                norm_b = np.linalg.norm(median_vector)
                if norm_a == 0 or norm_b == 0:
                    similarities[i] = 0.0
                else:
                    similarities[i] = (
                        np.dot(flat_weights[i], median_vector) / (norm_a * norm_b)
                    )

            # Kosinüs skorlarını normalize et (0-1)
            min_c, max_c = np.min(similarities), np.max(similarities)
            cos_norm = (
                np.ones(num_clients)
                if max_c == min_c
                else (similarities - min_c) / (max_c - min_c)
            )

            # Hibrit Skor = %50 Krum + %50 Kosinüs
            hybrid_scores = 0.5 * krum_norm + 0.5 * cos_norm

            # Eleme: en düşük hibrit skorlu num_malicious istemciyi çıkar.
            # Eğer en düşük skor, ortalamanın 1.5 std altındaysa kesin anomali say.
            num_to_select = max(1, num_clients - num_malicious)
            score_mean = np.mean(hybrid_scores)
            score_std  = np.std(hybrid_scores)
            anomaly_threshold = score_mean - 1.5 * score_std

            # Önce anomali eşiğinin altındakileri çıkar
            forced_out = [i for i in range(num_clients)
                          if hybrid_scores[i] < anomaly_threshold]
            # Kalan slotları en yüksek skorlularla doldur
            remaining = [i for i in range(num_clients) if i not in forced_out]
            remaining_sorted = sorted(remaining,
                                      key=lambda i: hybrid_scores[i], reverse=True)
            selected_indices = remaining_sorted[:num_to_select]

            print(f"  Krum Puanları (norm):       {np.round(krum_norm, 4)}")
            print(f"  Kosinüs Puanları (norm):    {np.round(cos_norm, 4)}")
            print(f"  Hibrit Skorları:            {np.round(hybrid_scores, 4)}")
            print(f"  Anomali eşiği (mean-1.5σ):  {anomaly_threshold:.4f}")
            print(f"  Zorla elinen indeksler:     {forced_out}")
            print(f"  Seçilen İstemci İndeksleri: {selected_indices}")

        else:
            print(f"Bilinmeyen yöntem: {self.robust_method}. Standart FedAvg uygulanıyor.")
            return super().aggregate_fit(server_round, results, failures)

        # ------------------------------------------------------------------ #
        #  Tespit İstatistiklerini Güncelle                                    #
        # ------------------------------------------------------------------ #
        # İstemcilerden gelen metadata'yı oku
        round_detail = {
            "round": server_round,
            "num_clients": num_clients,
            "selected_indices": list(selected_indices),
            "excluded_indices": [i for i in range(num_clients) if i not in selected_indices],
            "malicious_indices": [],       # gerçek kötü niyetli indeksler
            "correctly_excluded": [],      # doğru elenen kötü niyetliler
            "incorrectly_included": [],    # yanlış seçilen kötü niyetliler
            "flipped_samples_this_round": 0,
        }

        for idx, (_, fit_res) in enumerate(results):
            m = fit_res.metrics if fit_res.metrics else {}
            is_mal    = bool(m.get("is_malicious", 0))
            flipped   = int(m.get("flipped_count", 0))
            client_id = int(m.get("client_id", idx))

            if is_mal:
                round_detail["malicious_indices"].append(idx)
                round_detail["flipped_samples_this_round"] = flipped

                # flip sayısını SADECE ilk round'da kaydet (sabit değer, her round aynı)
                if not self.detection_stats["flipped_samples_recorded"]:
                    self.detection_stats["total_flipped_samples"] = flipped
                    self.detection_stats["flipped_samples_recorded"] = True

                self.detection_stats["total_malicious_rounds"] += 1

                if idx not in selected_indices:
                    round_detail["correctly_excluded"].append(idx)
                    self.detection_stats["correctly_excluded_rounds"] += 1
                    print(f"  [✓ TESPİT] İstemci #{client_id} (indeks {idx}) "
                          f"kötü niyetli olarak doğru elendi. "
                          f"({flipped} flip'li örnek engellendi)")
                else:
                    round_detail["incorrectly_included"].append(idx)
                    self.detection_stats["incorrectly_included_rounds"] += 1
                    print(f"  [✗ KAÇIRDI] İstemci #{client_id} (indeks {idx}) "
                          f"kötü niyetli ama seçildi! "
                          f"({flipped} flip'li örnek modele karıştı)")

        self.detection_stats["per_round"].append(round_detail)

        # Yalnızca seçilen istemcilerin sonuçlarıyla FedAvg uygula
        robust_results = [results[i] for i in selected_indices]
        print(f"  [{self.robust_method.upper()}] {len(robust_results)}/{num_clients} "
              f"istemci seçildi, birleştirme yapılıyor.")
        return super().aggregate_fit(server_round, robust_results, failures)


def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """İstemci örnek sayısına göre ağırlıklı ortalama accuracy hesaplar."""
    total_examples = sum(n for n, _ in metrics)
    accuracy = sum(n * m["accuracy"] for n, m in metrics) / total_examples
    print(f"\n--- [SUNUCU] Global Model Doğruluğu: %{accuracy * 100:.2f} ---\n")
    return {"accuracy": accuracy}


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Federatif Öğrenme Sunucusu")
    parser.add_argument("--num_clients", type=int, default=3,
                        help="Toplam istemci sayısı")
    parser.add_argument("--malicious_fraction", type=float, default=0.3,
                        help="Kötü niyetli istemci oranı (0.0 - 1.0)")
    parser.add_argument("--robust_method", type=str, default="hybrid",
                        choices=["cosine", "krum", "hybrid"],
                        help="Savunma algoritması: cosine | krum | hybrid")
    parser.add_argument("--num_rounds", type=int, default=20,
                        help="Federatif öğrenme tur sayısı")
    args = parser.parse_args()

    num_malicious = math.ceil(args.num_clients * args.malicious_fraction)
    num_normal = args.num_clients - num_malicious

    print(f"\n{'='*60}")
    print(f"Federatif Öğrenme Sunucusu Başlatılıyor")
    print(f"  Toplam İstemci  : {args.num_clients}")
    print(f"  Normal İstemci  : {num_normal}")
    print(f"  Kötü Niyetli    : {num_malicious}")
    print(f"  Savunma Yöntemi : {args.robust_method.upper()}")
    print(f"  Tur Sayısı      : {args.num_rounds}")
    print(f"{'='*60}\n")

    strategy = CustomRobustStrategy(
        robust_method=args.robust_method,
        malicious_fraction=args.malicious_fraction,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    # Sonuçları kaydet
    total_mal_rounds   = strategy.detection_stats["total_malicious_rounds"]
    correctly_excluded = strategy.detection_stats["correctly_excluded_rounds"]
    detection_rate     = (
        correctly_excluded / total_mal_rounds * 100
        if total_mal_rounds > 0 else 0.0
    )

    results_data = {
        "method": args.robust_method,
        "num_clients": args.num_clients,
        "num_normal": num_normal,
        "num_malicious": num_malicious,
        "num_rounds": args.num_rounds,
        "losses": history.losses_distributed,
        "accuracies": history.metrics_distributed.get("accuracy", []),
        # ---- Label Flipping Tespit İstatistikleri ----
        "label_flipping": {
            "total_flipped_samples":       strategy.detection_stats["total_flipped_samples"],
            "total_malicious_rounds":      total_mal_rounds,
            "correctly_excluded_rounds":   correctly_excluded,
            "incorrectly_included_rounds": strategy.detection_stats["incorrectly_included_rounds"],
            "detection_rate_pct":          round(detection_rate, 2),
            "per_round":                   strategy.detection_stats["per_round"],
        },
    }
    with open("results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print("\nSonuçlar 'results.json' dosyasına kaydedildi.")
    print(f"\n{'='*60}")
    print(f"  Label Flipping Tespit Özeti")
    print(f"{'='*60}")
    print(f"  Toplam flip'li örnek (tüm roundlar) : {strategy.detection_stats['total_flipped_samples']}")
    print(f"  Kötü niyetli katılım (round sayısı) : {total_mal_rounds}")
    print(f"  Doğru eleme sayısı                  : {correctly_excluded}")
    print(f"  Yanlış seçim sayısı                 : {strategy.detection_stats['incorrectly_included_rounds']}")
    print(f"  Tespit Başarı Oranı                 : %{detection_rate:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
