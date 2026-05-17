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

    Standart FedAvg'ın üzerine inşa edilmiş beş savunma yöntemi sunar:
    - krum:     Öklid uzaklığına dayalı anomali tespiti (Blanchard et al., 2017)
    - cosine:   Kosinüs benzerliğine dayalı yön anomalisi tespiti
    - hybrid:   Krum + Kosinüs skorlarının ağırlıklı ortalaması + EMA itibar takibi
    - fltrust:  Sunucu referans gradyanına dayalı güven skoru (Wang et al., 2021)
    - ensemble: Krum + Kosinüs + FLTrust + EMA'nın tamamını birleştirir (en güçlü)

    Her yöntem, çoğunluktan sapan (zehirli) gradyanları dışlayarak
    yalnızca güvenilir istemcilerin güncellemelerini birleştirir.
    """

    def __init__(self, robust_method: str = "hybrid",
                 malicious_fraction: float = 0.3,
                 ema_alpha: float = 0.15,
                 server_model=None,
                 server_data=None, **kwargs):
        super().__init__(**kwargs)
        self.robust_method = robust_method
        self.malicious_fraction = malicious_fraction
        self.ema_alpha = ema_alpha
        
        # FLTrust için sunucu modeli ve temiz veri seti
        self.server_model = server_model
        self.server_data = server_data  # (x_train, y_train) tuple

        self.client_prev_weights: dict = {}
        # client_id → EMA güven skoru (0-1)
        self.client_ema: dict = {}
        # client_id → kaç roundda görüldü
        self.client_round_count: dict = {}
        # Sunucunun bir önceki rounddaki global ağırlıkları (delta hesabı için)
        self.global_weights: Optional[List[np.ndarray]] = None
        # Bir önceki round'un global güncelleme yönü (momentum için)
        self.prev_global_delta: Optional[np.ndarray] = None

        self.detection_stats = {
            "total_flipped_samples":       0,
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

        # Gradyan norm kırpma: Her istemcinin delta normunu medyan norma kırp.
        # Saldırgan büyük normlu gradyanla çoğunluğa "yakın" görünebilir.
        # Norm kırpma, tüm deltaları eşit büyüklüğe getirip sadece YÖN farkına
        # odaklanmamızı sağlar.
        delta_norms = np.array([np.linalg.norm(d) for d in deltas])
        median_norm = np.median(delta_norms)
        if median_norm > 0:
            for i in range(len(deltas)):
                if delta_norms[i] > median_norm:
                    deltas[i] = deltas[i] * (median_norm / delta_norms[i])

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
        #  FLTRUST (Güven Skoru Bazlı Savunma)                               #
        # ------------------------------------------------------------------ #
        elif self.robust_method == "fltrust":
            print(f"\n--- [Round {server_round}] FLTrust Algoritması Uygulanıyor ---")

            if self.server_model is None or self.server_data is None:
                print("[HATA] FLTrust için sunucu modeli ve temiz veri seti gerekli!")
                return super().aggregate_fit(server_round, results, failures)

            # İlk round'da global ağırlık yoksa ilk istemciden al
            if self.global_weights is None:
                self.global_weights = [w.copy() for w in client_weights[0]]

            # ---- Sunucu referans deltasını hesapla ----
            # Sunucu kendi temiz verisiyle 1 epoch eğitim yapar,
            # eski ağırlıklar ile yeni ağırlıklar arasındaki fark = referans delta.
            # Bu delta "doğru yön"ü temsil eder.
            old_weights_flat = flatten_weights(self.global_weights)
            
            # Modelin build edildiğinden emin ol (ilk round'da gerekebilir)
            x_server, y_server = self.server_data
            if len(self.server_model.weights) == 0:
                dummy = np.zeros((1, x_server.shape[1]), dtype=np.int32)
                self.server_model(dummy, training=False)
            
            self.server_model.set_weights(self.global_weights)
            self.server_model.fit(x_server, y_server, epochs=1, batch_size=32, verbose=0)
            new_weights_flat = flatten_weights(self.server_model.get_weights())
            server_delta = new_weights_flat - old_weights_flat
            server_norm  = np.linalg.norm(server_delta)

            if server_norm == 0:
                print("[UYARI] Sunucu referans deltası sıfır, FedAvg'a düşülüyor.")
                return super().aggregate_fit(server_round, results, failures)

            # ---- Her istemci için Trust Score hesapla ----
            # TS = ReLU( cos(client_delta, server_delta) )
            # Label flipping saldırısı sunucu referansına ters yönde gider
            # → negatif kosinüs → ReLU ile 0'a indirilir → ağırlık = 0 → elenir.
            trust_scores = np.zeros(num_clients)
            for i in range(num_clients):
                client_norm = np.linalg.norm(deltas[i])
                if client_norm > 0:
                    cos_sim = np.dot(deltas[i], server_delta) / (client_norm * server_norm)
                    trust_scores[i] = max(0.0, float(cos_sim))

            total_trust = np.sum(trust_scores)
            if total_trust == 0:
                print("[UYARI] Tüm istemciler negatif TS aldı! FedAvg'a düşülüyor.")
                return super().aggregate_fit(server_round, results, failures)

            # ---- Trust Score ağırlıklı model birleştirme ----
            normalized_trust = trust_scores / total_trust
            aggregated_weights = []
            for layer_idx in range(len(client_weights[0])):
                layer = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
                for i in range(num_clients):
                    layer += normalized_trust[i] * client_weights[i][layer_idx]
                aggregated_weights.append(layer)

            selected_indices  = [i for i in range(num_clients) if trust_scores[i] > 0]
            excluded_indices  = [i for i in range(num_clients) if trust_scores[i] <= 0]

            print(f"  Sunucu Referans Norm:       {server_norm:.6f}")
            print(f"  Güven Skorları (TS):        {np.round(trust_scores, 4)}")
            print(f"  Normalize Ağırlıklar:       {np.round(normalized_trust, 4)}")
            print(f"  Elenen İstemciler (TS≤0):   {excluded_indices}")
            print(f"  Seçilen İstemci İndeksleri: {selected_indices}")

            # ---- Tespit istatistiklerini güncelle ----
            round_client_ids = []
            for _, fit_res in results:
                m = fit_res.metrics if fit_res.metrics else {}
                round_client_ids.append(int(m.get("client_id", -1)))

            round_detail = {
                "round":                      server_round,
                "num_clients":                num_clients,
                "selected_indices":           list(selected_indices),
                "excluded_indices":           excluded_indices,
                "malicious_client_ids":       [],
                "malicious_indices":          [],
                "correctly_excluded":         [],
                "incorrectly_included":       [],
                "flipped_samples_this_round": 0,
            }

            for idx, (_, fit_res) in enumerate(results):
                m = fit_res.metrics if fit_res.metrics else {}
                if not bool(m.get("is_malicious", 0)):
                    continue
                cid     = int(m.get("client_id", idx))
                flipped = int(m.get("flipped_count", 0))

                round_detail["malicious_client_ids"].append(cid)
                round_detail["malicious_indices"].append(idx)
                # Bu round'daki toplam flip = tüm kötü niyetlilerin flip sayısı toplamı
                round_detail["flipped_samples_this_round"] += flipped

                # Toplam flip sayısını biriktir (birden fazla kötü niyetli için doğru toplam)
                self.detection_stats["total_flipped_samples"] += flipped

                self.detection_stats["total_malicious_rounds"] += 1

                if idx not in selected_indices:
                    round_detail["correctly_excluded"].append(cid)
                    self.detection_stats["correctly_excluded_rounds"] += 1
                    print(f"  [✓ TESPİT] İstemci #{cid} (results[{idx}]) "
                          f"doğru elendi (TS={trust_scores[idx]:.4f}). "
                          f"({flipped} flip'li örnek engellendi)")
                else:
                    round_detail["incorrectly_included"].append(cid)
                    self.detection_stats["incorrectly_included_rounds"] += 1
                    print(f"  [✗ KAÇIRDI] İstemci #{cid} (results[{idx}]) "
                          f"seçildi (TS={trust_scores[idx]:.4f})! "
                          f"({flipped} flip'li örnek modele karıştı)")

            self.detection_stats["per_round"].append(round_detail)

            # Global ağırlıkları güncelle ve sonucu döndür
            self.global_weights = aggregated_weights
            from flwr.common import ndarrays_to_parameters
            aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
            print(f"  [FLTRUST] {len(selected_indices)}/{num_clients} istemci seçildi.")
            return aggregated_parameters, {}

        # ------------------------------------------------------------------ #
        #  HİBRİT (KRUM + COSİNÜS + FLTRUST + EMA)                           #
        # ------------------------------------------------------------------ #
        elif self.robust_method == "hybrid":
            print(f"\n--- [Round {server_round}] Hibrit Savunma (Krum + Kosinüs + FLTrust + EMA) Uygulanıyor ---")
            check_krum_feasibility(num_clients, num_malicious)

            round_client_ids = []
            for _, fit_res in results:
                m = fit_res.metrics if fit_res.metrics else {}
                round_client_ids.append(int(m.get("client_id", -1)))

            # -- 1. Krum skoru --
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
                np.ones(num_clients) if max_k == min_k
                else 1.0 - (krum_scores - min_k) / (max_k - min_k)
            )

            # -- 2. Kosinüs skoru (medyan referans) --
            median_delta = np.median(deltas, axis=0)
            similarities = np.zeros(num_clients)
            for i in range(num_clients):
                norm_a = np.linalg.norm(deltas[i])
                norm_b = np.linalg.norm(median_delta)
                if norm_a == 0 or norm_b == 0:
                    similarities[i] = 0.0
                else:
                    similarities[i] = np.dot(deltas[i], median_delta) / (norm_a * norm_b)
            min_c, max_c = np.min(similarities), np.max(similarities)
            cos_norm = (
                np.ones(num_clients) if max_c == min_c
                else (similarities - min_c) / (max_c - min_c)
            )

            # -- 3. FLTrust skoru (sunucu referans gradyanı) --
            # Sunucu kendi temiz verisiyle "doğru yön"ü hesaplar.
            # İstemci güncellemeleri bu referansla karşılaştırılır.
            # Sunucu modeli/verisi yoksa nötr skor (0.5) atanır → ağırlıklandırmaya etkisi sınırlı.
            fltrust_norm = np.ones(num_clients) * 0.5
            if self.server_model is not None and self.server_data is not None:
                ref_weights = self.global_weights if self.global_weights is not None else client_weights[0]
                old_flat = flatten_weights(ref_weights)
                x_server, y_server = self.server_data

                if len(self.server_model.weights) == 0:
                    dummy = np.zeros((1, x_server.shape[1]), dtype=np.int32)
                    self.server_model(dummy, training=False)

                self.server_model.set_weights(ref_weights)
                self.server_model.fit(x_server, y_server, epochs=1, batch_size=32, verbose=0)
                new_flat = flatten_weights(self.server_model.get_weights())
                server_delta = new_flat - old_flat
                server_norm  = np.linalg.norm(server_delta)

                if server_norm > 0:
                    raw_ts = np.zeros(num_clients)
                    for i in range(num_clients):
                        cn = np.linalg.norm(deltas[i])
                        if cn > 0:
                            raw_ts[i] = max(0.0, float(
                                np.dot(deltas[i], server_delta) / (cn * server_norm)
                            ))
                    min_ts, max_ts = np.min(raw_ts), np.max(raw_ts)
                    fltrust_norm = (
                        np.ones(num_clients) * 0.5 if max_ts == min_ts
                        else (raw_ts - min_ts) / (max_ts - min_ts)
                    )
                    print(f"  FLTrust Ham TS:             {np.round(raw_ts, 4)}")
                    print(f"  FLTrust Norm:               {np.round(fltrust_norm, 4)}")
                else:
                    print("  [UYARI] Sunucu referans deltası sıfır, FLTrust nötr skora düşüldü.")
            else:
                print("  [BİLGİ] Sunucu modeli/verisi yok, FLTrust bileşeni nötr (0.5).")

            # -- 4. Anlık hibrit skor: Krum %25 + Kosinüs %35 + FLTrust %40 --
            # FLTrust en güvenilir sinyal (sunucu referansı) → en yüksek ağırlık.
            # Sunucu verisi yoksa Krum+Kosinüs ağırlıkları devreye girer.
            instant_scores = 0.25 * krum_norm + 0.35 * cos_norm + 0.40 * fltrust_norm

            # -- 5. EMA güncelle --
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

            # -- 6. Birleşik sıralama skoru --
            # İlk 3 round'da anlık skora güven (EMA henüz olgunlaşmadı),
            # sonrasında EMA ağırlığını artır.
            num_to_select = max(1, num_clients - num_malicious)
            combined = np.zeros(num_clients)
            for i, cid in enumerate(round_client_ids):
                n = self.client_round_count.get(cid, 1)
                ema_val = self.client_ema.get(cid, instant_scores[i])
                if n <= 3:
                    combined[i] = instant_scores[i]
                else:
                    combined[i] = 0.30 * instant_scores[i] + 0.70 * ema_val

            selected_indices = np.argsort(combined)[-num_to_select:].tolist()
            forced_out = [i for i in range(num_clients) if i not in selected_indices]

            self.prev_global_delta = median_delta.copy()

            ema_display = {
                round_client_ids[i]: round(self.client_ema[round_client_ids[i]], 4)
                for i in range(num_clients)
            }
            print(f"  Krum Puanları (norm):       {np.round(krum_norm, 4)}")
            print(f"  Kosinüs Puanları (norm):    {np.round(cos_norm, 4)}")
            print(f"  Anlık Hibrit Skorları:      {np.round(instant_scores, 4)}")
            print(f"  EMA Güven Skorları:         {ema_display}")
            print(f"  Birleşik Sıralama Skoru:    {np.round(combined, 4)}")
            print(f"  Elinen indeksler:           {forced_out}")
            print(f"  Seçilen İstemci İndeksleri: {selected_indices}")

        else:
            print(f"Bilinmeyen yöntem: {self.robust_method}. Standart FedAvg uygulanıyor.")
            return super().aggregate_fit(server_round, results, failures)                                    #
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
            # Bu round'daki toplam flip = tüm kötü niyetlilerin flip sayısı toplamı
            round_detail["flipped_samples_this_round"] += flipped

            # Toplam flip sayısını biriktir (birden fazla kötü niyetli için doğru toplam)
            self.detection_stats["total_flipped_samples"] += flipped

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
    import os
    import numpy as np
    import pandas as pd
    from src.bilstm_model import build_bilstm_model
    from src.glove_loader import load_glove_embeddings
    from src.data_preprocessing import (
        load_global_tokenizer,
        texts_to_padded,
    )

    parser = argparse.ArgumentParser(description="Federatif Öğrenme Sunucusu")
    parser.add_argument("--num_clients", type=int, default=3,
                        help="Toplam istemci sayısı")
    parser.add_argument("--malicious_fraction", type=float, default=0.3,
                        help="Kötü niyetli istemci oranı (0.0 - 1.0)")
    parser.add_argument("--robust_method", type=str, default="hybrid",
                        choices=["cosine", "krum", "hybrid", "fltrust"],
                        help="Savunma algoritması: cosine | krum | hybrid | fltrust")
    parser.add_argument("--num_rounds", type=int, default=20,
                        help="Federatif öğrenme tur sayısı")
    parser.add_argument("--data_path", type=str, default="email_text.csv",
                        help="CSV veri seti yolu (FLTrust için sunucu temiz verisi)")
    parser.add_argument("--glove_path", type=str, default="glove.6B.100d.txt",
                        help="GloVe dosyası yolu")
    parser.add_argument("--server_data_size", type=int, default=100,
                        help="FLTrust için sunucu temiz veri seti boyutu")
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

    # FLTrust için sunucu modeli ve temiz veri seti hazırla
    server_model = None
    server_data = None
    
    if args.robust_method in ("fltrust", "hybrid"):
        print("[FLTrust] Sunucu temiz veri seti hazırlanıyor...")
        
        # Veriyi yükle
        if os.path.exists(args.data_path):
            df = pd.read_csv(args.data_path)
            texts = df['text'].astype(str).tolist()
            labels = np.array(df['label'].astype(int).tolist())
        else:
            print(f"HATA: '{args.data_path}' bulunamadı!")
            return
        
        # Tokenizer yükle
        tokenizer = load_global_tokenizer("tokenizer.pkl")
        max_length = 50
        max_words = 5000
        
        # Sunucu için küçük temiz veri seti oluştur (dengeli)
        # Her sınıftan eşit sayıda örnek al
        phishing_indices = np.where(labels == 1)[0]
        normal_indices = np.where(labels == 0)[0]
        
        samples_per_class = args.server_data_size // 2
        selected_phishing = np.random.choice(phishing_indices, samples_per_class, replace=False)
        selected_normal = np.random.choice(normal_indices, samples_per_class, replace=False)
        selected_indices = np.concatenate([selected_phishing, selected_normal])
        np.random.shuffle(selected_indices)
        
        server_texts = [texts[i] for i in selected_indices]
        server_labels = labels[selected_indices]
        
        x_server = texts_to_padded(server_texts, tokenizer, max_length=max_length)
        y_server = server_labels
        
        print(f"  Sunucu veri seti: {len(x_server)} örnek "
              f"(Phishing: {np.sum(y_server)}, Normal: {len(y_server) - np.sum(y_server)})")
        
        # Sunucu modeli oluştur
        vocab_size = min(max_words, len(tokenizer.word_index) + 1)
        embedding_matrix = None
        if os.path.exists(args.glove_path):
            embedding_matrix = load_glove_embeddings(
                args.glove_path, tokenizer.word_index, vocab_size, embedding_dim=100
            )
        
        server_model = build_bilstm_model(
            vocab_size=vocab_size,
            embedding_dim=100,
            max_length=max_length,
            embedding_matrix=embedding_matrix
        )
        
        # Modeli build et: set_weights() çağrılabilmesi için
        # en az bir forward pass gerekir (Keras lazy build)
        dummy_input = np.zeros((1, max_length), dtype=np.int32)
        server_model(dummy_input, training=False)
        print(f"  Sunucu modeli build edildi: {server_model.count_params()} parametre")
        
        server_data = (x_server, y_server)
        print("[FLTrust] Sunucu hazırlığı tamamlandı.\n")

    strategy = CustomRobustStrategy(
        robust_method=args.robust_method,
        malicious_fraction=args.malicious_fraction,
        server_model=server_model,
        server_data=server_data,
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
                                            for k, v in strategy.client_ema.items()}
                                            if args.robust_method == "hybrid" else {},
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
