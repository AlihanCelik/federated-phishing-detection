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
                 malicious_fraction: float = 0.3,
                 ema_alpha: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.robust_method = robust_method
        self.malicious_fraction = malicious_fraction
        self.ema_alpha = ema_alpha

        self.client_prev_weights: dict = {}
        # client_id → son round'daki delta vektörü (tutarsızlık hesabı için)
        self.client_prev_deltas: dict = {}
        # client_id → EMA güven skoru (0-1)
        self.client_ema: dict = {}
        # client_id → kaç roundda görüldü
        self.client_round_count: dict = {}
        # Sunucunun bir önceki rounddaki global ağırlıkları (delta hesabı için)
        self.global_weights: Optional[List[np.ndarray]] = None

        self.detection_stats = {
            "total_flipped_samples":       0,
            "flipped_samples_recorded":    False,
            "total_malicious_rounds":      0,
            "correctly_excluded_rounds":   0,
            "incorrectly_included_rounds": 0,
            "per_round": [],
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

        # Delta'ları (güncellemeleri) hesapla.
        # Ağırlıkların kendisi yerine değişim yönüne (delta) bakmak,
        # label-flipping gibi saldırıları tespit etmede çok daha etkilidir.
        if self.global_weights is not None:
            flat_global = flatten_weights(self.global_weights)
            deltas = np.array([w - flat_global for w in flat_weights])
        else:
            # İlk round'da global ağırlık henüz yoksa, medyanı referans al
            median_w = np.median(flat_weights, axis=0)
            deltas = np.array([w - median_w for w in flat_weights])

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

            # Referans vektör: tüm güncellemelerin (deltaların) ortalaması
            mean_delta = np.mean(deltas, axis=0)

            similarities = np.zeros(num_clients)
            for i in range(num_clients):
                norm_a = np.linalg.norm(deltas[i])
                norm_b = np.linalg.norm(mean_delta)
                if norm_a == 0 or norm_b == 0:
                    similarities[i] = 0.0
                else:
                    similarities[i] = (
                        np.dot(deltas[i], mean_delta) / (norm_a * norm_b)
                    )

            # En yüksek benzerliğe sahip (num_clients - num_malicious) istemciyi seç
            num_to_select = max(1, num_clients - num_malicious)
            selected_indices = np.argsort(similarities)[-num_to_select:].tolist()

            print(f"  Kosinüs Benzerlikleri:      {np.round(similarities, 4)}")
            print(f"  Seçilen İstemci İndeksleri: {selected_indices}")

        # ------------------------------------------------------------------ #
        #  HİBRİT (KRUM + COSİNÜS + EMA HAFIZA)                              #
        # ------------------------------------------------------------------ #
        elif self.robust_method == "hybrid":
            print(f"\n--- [Round {server_round}] Hibrit Savunma (Krum + Kosinüs + EMA) Uygulanıyor ---")
            check_krum_feasibility(num_clients, num_malicious)

            # Bu round'daki her results[i] için client_id'yi al
            round_client_ids = []
            for _, fit_res in results:
                m = fit_res.metrics if fit_res.metrics else {}
                round_client_ids.append(int(m.get("client_id", -1)))

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

            min_k, max_k = np.min(krum_scores), np.max(krum_scores)
            krum_norm = (
                np.ones(num_clients)
                if max_k == min_k
                else 1.0 - (krum_scores - min_k) / (max_k - min_k)
            )

            # -- Kosinüs hesaplama (medyan referans) --
            median_delta = np.median(deltas, axis=0)
            similarities = np.zeros(num_clients)
            for i in range(num_clients):
                norm_a = np.linalg.norm(deltas[i])
                norm_b = np.linalg.norm(median_delta)
                if norm_a == 0 or norm_b == 0:
                    similarities[i] = 0.0
                else:
                    similarities[i] = (
                        np.dot(deltas[i], median_delta) / (norm_a * norm_b)
                    )

            min_c, max_c = np.min(similarities), np.max(similarities)
            cos_norm = (
                np.ones(num_clients)
                if max_c == min_c
                else (similarities - min_c) / (max_c - min_c)
            )

            # -- Round-to-round tutarsızlık skoru --
            # İstemcinin mevcut güncelleme yönü (delta) ile önceki rounddaki yönü (delta)
            # arasındaki kosinüs benzerliği. Normalde istemciler IID verilerde benzer
            # yönlere giderler, ancak ataklar genellikle keskin ve tutarsız yön değişimleri yaratabilir.
            consistency_scores = np.ones(num_clients)  # varsayılan: tutarlı
            for i, cid in enumerate(round_client_ids):
                if cid in self.client_prev_deltas:
                    prev = self.client_prev_deltas[cid]
                    curr = deltas[i]
                    norm_p = np.linalg.norm(prev)
                    norm_c = np.linalg.norm(curr)
                    if norm_p > 0 and norm_c > 0:
                        consistency_scores[i] = float(
                            np.dot(prev, curr) / (norm_p * norm_c)
                        )
                        # [-1, 1] → [0, 1] aralığına taşı
                        consistency_scores[i] = (consistency_scores[i] + 1.0) / 2.0

            # Tutarsızlık skorlarını normalize et
            min_cs, max_cs = np.min(consistency_scores), np.max(consistency_scores)
            cons_norm = (
                np.ones(num_clients)
                if max_cs == min_cs
                else (consistency_scores - min_cs) / (max_cs - min_cs)
            )

            # ÖNEMLİ DÜZELTME: Label Flipping saldırılarında saldırgan her zaman 
            # aynı sahte hedefe (zehirli minimuma) doğru çekim yapar. Bu yüzden 
            # ardışık roundlarda güncelleme yönü aşırı derecede "TUTARLI" çıkar. 
            # Normal istemciler ise stokastik (rastgele) mini-batchler kullandığı 
            # için yönleri dalgalanır. Yani aşırı tutarlılık bir anomali (saldırı) 
            # belirtisidir! Bu yüzden Tutarsızlık skorunu tersine çeviriyoruz:
            # Ne kadar aşırı tutarlıysa (1.0), o kadar düşük puan (0.0) almalı.
            cons_norm = 1.0 - cons_norm

            # -- Anlık birleşik skor: Krum %35 + Kosinüs %35 + Tutarsızlık %30 --
            instant_scores = 0.35 * krum_norm + 0.35 * cos_norm + 0.30 * cons_norm

            # -- EMA güncelle --
            for i, cid in enumerate(round_client_ids):
                if cid not in self.client_ema:
                    self.client_ema[cid] = instant_scores[i]
                    self.client_round_count[cid] = 1
                else:
                    self.client_ema[cid] = (
                        self.ema_alpha * instant_scores[i]
                        + (1 - self.ema_alpha) * self.client_ema[cid]
                    )
                    self.client_round_count[cid] += 1

            # -- Nihai skor: az geçmişte anlık ağır, çok geçmişte EMA ağır --
            final_scores = np.zeros(num_clients)
            for i, cid in enumerate(round_client_ids):
                n = self.client_round_count[cid]
                if n <= 2:
                    final_scores[i] = instant_scores[i]
                else:
                    final_scores[i] = 0.35 * instant_scores[i] + 0.65 * self.client_ema[cid]

            # -- Eleme: her zaman en düşük num_malicious istemciyi çıkar (sıralama bazlı) --
            # Eşik yaklaşımı kötü niyetli yüksek skor aldığında işe yaramaz.
            # Sıralama bazlı eleme garantili olarak num_malicious istemciyi dışarıda bırakır.
            num_to_select = max(1, num_clients - num_malicious)

            # Sıralama skoru: final_score'a EMA'yı da ekle (daha stabil)
            combined = np.zeros(num_clients)
            for i, cid in enumerate(round_client_ids):
                ema_val = self.client_ema.get(cid, instant_scores[i])
                combined[i] = 0.5 * final_scores[i] + 0.5 * ema_val

            # En yüksek num_to_select istemciyi seç
            selected_indices = np.argsort(combined)[-num_to_select:].tolist()
            forced_out = [i for i in range(num_clients) if i not in selected_indices]

            # -- Ağırlıkları/Deltaları bir sonraki round için sakla --
            for i, cid in enumerate(round_client_ids):
                self.client_prev_weights[cid] = flat_weights[i].copy()
                self.client_prev_deltas[cid] = deltas[i].copy()

            ema_display = {round_client_ids[i]: round(self.client_ema[round_client_ids[i]], 4)
                           for i in range(num_clients)}
            print(f"  Krum Puanları (norm):       {np.round(krum_norm, 4)}")
            print(f"  Kosinüs Puanları (norm):    {np.round(cos_norm, 4)}")
            print(f"  Tutarsızlık Puanları (norm):{np.round(cons_norm, 4)}")
            print(f"  Anlık Hibrit Skorları:      {np.round(instant_scores, 4)}")
            print(f"  EMA Güven Skorları:         {ema_display}")
            print(f"  Nihai Skorlar:              {np.round(final_scores, 4)}")
            print(f"  Birleşik Sıralama Skoru:    {np.round(combined, 4)}")
            print(f"  Elinen indeksler:           {forced_out}")
            print(f"  Seçilen İstemci İndeksleri: {selected_indices}")

        else:
            print(f"Bilinmeyen yöntem: {self.robust_method}. Standart FedAvg uygulanıyor.")
            return super().aggregate_fit(server_round, results, failures)

        # ------------------------------------------------------------------ #
        #  Tespit İstatistiklerini Güncelle                                    #
        # ------------------------------------------------------------------ #
        # İstemcilerden gelen metadata'yı oku.
        # ÖNEMLİ: results listesindeki sıra (idx) her round'da değişebilir
        # çünkü Flower istemcileri farklı sırada bağlanabilir. Bu nedenle
        # "kötü niyetli mi?" sorusunu idx üzerinden değil, istemcinin
        # gönderdiği client_id metadata'sı üzerinden yanıtlıyoruz.
        # selected_indices de aynı results sıralamasına göre hesaplandığından
        # "bu istemci seçildi mi?" kontrolü idx ile yapılmaya devam eder.

        # Adım 1: Her results[idx] için client_id → is_malicious eşlemesini kur
        client_meta = {}   # idx -> {"client_id", "is_malicious", "flipped_count"}
        for idx, (_, fit_res) in enumerate(results):
            m = fit_res.metrics if fit_res.metrics else {}
            client_meta[idx] = {
                "client_id":    int(m.get("client_id", idx)),
                "is_malicious": bool(m.get("is_malicious", 0)),
                "flipped":      int(m.get("flipped_count", 0)),
            }

        round_detail = {
            "round": server_round,
            "num_clients": num_clients,
            "selected_indices": list(selected_indices),
            "excluded_indices": [i for i in range(num_clients) if i not in selected_indices],
            "malicious_client_ids": [],    # gerçek kötü niyetli istemci ID'leri (sabit)
            "malicious_indices": [],       # bu round'daki results listesindeki konumları
            "correctly_excluded": [],      # doğru elenen kötü niyetlilerin client_id'leri
            "incorrectly_included": [],    # yanlış seçilen kötü niyetlilerin client_id'leri
            "flipped_samples_this_round": 0,
        }

        for idx, meta in client_meta.items():
            if not meta["is_malicious"]:
                continue

            client_id = meta["client_id"]
            flipped   = meta["flipped"]

            round_detail["malicious_client_ids"].append(client_id)
            round_detail["malicious_indices"].append(idx)
            round_detail["flipped_samples_this_round"] = flipped

            # flip sayısını her kötü niyetli istemci için bir kez kaydet
            # (client_id bazında takip et — aynı istemcinin değerini tekrar yazma)
            if not self.detection_stats["flipped_samples_recorded"]:
                self.detection_stats["total_flipped_samples"] = flipped
                self.detection_stats["flipped_samples_recorded"] = True
            else:
                # Birden fazla kötü niyetli varsa en büyük flip sayısını tut
                if flipped > self.detection_stats["total_flipped_samples"]:
                    self.detection_stats["total_flipped_samples"] = flipped

            self.detection_stats["total_malicious_rounds"] += 1

            # idx, savunma algoritmasının kullandığı results sıralamasıyla aynı
            # olduğundan seçim kontrolü idx üzerinden yapılır
            if idx not in selected_indices:
                round_detail["correctly_excluded"].append(client_id)
                self.detection_stats["correctly_excluded_rounds"] += 1
                print(f"  [✓ TESPİT] İstemci #{client_id} (results[{idx}]) "
                      f"kötü niyetli olarak doğru elendi. "
                      f"({flipped} flip'li örnek engellendi)")
            else:
                round_detail["incorrectly_included"].append(client_id)
                self.detection_stats["incorrectly_included_rounds"] += 1
                print(f"  [✗ KAÇIRDI] İstemci #{client_id} (results[{idx}]) "
                      f"kötü niyetli ama seçildi! "
                      f"({flipped} flip'li örnek modele karıştı)")

        self.detection_stats["per_round"].append(round_detail)

        # Yalnızca seçilen istemcilerin sonuçlarıyla FedAvg uygula
        robust_results = [results[i] for i in selected_indices]
        print(f"  [{self.robust_method.upper()}] {len(robust_results)}/{num_clients} "
              f"istemci seçildi, birleştirme yapılıyor.")
        
        aggregated_parameters, metrics = super().aggregate_fit(server_round, robust_results, failures)
        
        # Gelecek round'da delta hesaplayabilmek için güncellenmiş global ağırlıkları sakla
        if aggregated_parameters is not None:
            self.global_weights = parameters_to_ndarrays(aggregated_parameters)
            
        return aggregated_parameters, metrics


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
            "ema_final_scores":            {str(k): round(v, 4)
                                            for k, v in strategy.client_ema.items()},
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
