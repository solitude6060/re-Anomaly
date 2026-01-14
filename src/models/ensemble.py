from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class EnsembleAnomalyDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.heads: dict[str, nn.Module] = {}
        self.weights: dict[str, float] = config.get(
            "weights", {"patchcore": 0.5, "fastflow": 0.3, "rectflow": 0.2}
        )
        self.normalize_scores = config.get("normalize_scores", True)
        self._fitted = False

    def add_head(self, name: str, head: nn.Module, weight: float | None = None) -> None:
        self.heads[name] = head
        if weight is not None:
            self.weights[name] = weight

    def fit(self, features: list[torch.Tensor]) -> None:
        for name, head in self.heads.items():
            if hasattr(head, "fit"):
                head.fit(features)
        self._fitted = True

    def predict(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        all_scores = {}
        all_maps = {}

        for name, head in self.heads.items():
            head.eval()
            with torch.no_grad():
                output = head(features)
                all_scores[name] = output["anomaly_score"]
                if "anomaly_map" in output:
                    all_maps[name] = output["anomaly_map"]

        if self.normalize_scores:
            normalized_scores = {}
            for name, scores in all_scores.items():
                min_val = scores.min()
                max_val = scores.max()
                if max_val - min_val > 1e-8:
                    normalized_scores[name] = (scores - min_val) / (max_val - min_val)
                else:
                    normalized_scores[name] = torch.zeros_like(scores)
            all_scores = normalized_scores

        total_weight = sum(self.weights.get(name, 0) for name in self.heads.keys())
        ensemble_score = torch.zeros_like(list(all_scores.values())[0])

        for name, scores in all_scores.items():
            weight = self.weights.get(name, 1.0 / len(self.heads))
            ensemble_score += weight * scores / total_weight

        return {
            "anomaly_score": ensemble_score,
            "individual_scores": all_scores,
            "individual_maps": all_maps,
        }


class AdaptiveEnsemble:
    def __init__(self, heads: dict[str, nn.Module]) -> None:
        self.heads = heads
        self.weights: dict[str, float] = {name: 1.0 / len(heads) for name in heads}
        self._score_stats: dict[str, tuple[float, float]] = {}

    def calibrate(
        self,
        features_list: list[list[torch.Tensor]],
        labels: np.ndarray,
        val_split: float = 0.2,
    ) -> dict[str, float]:
        n_samples = len(features_list)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]

        val_features = [features_list[i] for i in val_indices]
        val_labels = labels[val_indices]

        head_scores = {}
        head_aurocs = {}

        for name, head in self.heads.items():
            scores = []
            head.eval()
            with torch.no_grad():
                for features in val_features:
                    output = head(features)
                    scores.append(output["anomaly_score"].cpu().numpy())

            scores = np.concatenate(scores)
            self._score_stats[name] = (float(scores.mean()), float(scores.std()))
            head_scores[name] = scores

            if len(np.unique(val_labels)) > 1:
                auroc = roc_auc_score(val_labels, scores)
                head_aurocs[name] = auroc
            else:
                head_aurocs[name] = 0.5

        total_auroc = sum(head_aurocs.values())
        if total_auroc > 0:
            self.weights = {
                name: auroc / total_auroc for name, auroc in head_aurocs.items()
            }
        else:
            self.weights = {name: 1.0 / len(self.heads) for name in self.heads}

        return head_aurocs

    def predict(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        all_scores = {}

        for name, head in self.heads.items():
            head.eval()
            with torch.no_grad():
                output = head(features)
                scores = output["anomaly_score"]

                if name in self._score_stats:
                    mean, std = self._score_stats[name]
                    if std > 1e-8:
                        scores = (scores - mean) / std

                all_scores[name] = scores

        ensemble_score = torch.zeros_like(list(all_scores.values())[0])
        for name, scores in all_scores.items():
            ensemble_score += self.weights[name] * scores

        return {
            "anomaly_score": ensemble_score,
            "individual_scores": all_scores,
            "weights": self.weights,
        }


class VotingEnsemble:
    def __init__(
        self, heads: dict[str, nn.Module], threshold_percentile: float = 95
    ) -> None:
        self.heads = heads
        self.threshold_percentile = threshold_percentile
        self.thresholds: dict[str, float] = {}

    def fit_thresholds(self, features_list: list[list[torch.Tensor]]) -> None:
        for name, head in self.heads.items():
            scores = []
            head.eval()
            with torch.no_grad():
                for features in features_list:
                    output = head(features)
                    scores.append(output["anomaly_score"].cpu().numpy())

            all_scores = np.concatenate(scores)
            self.thresholds[name] = float(
                np.percentile(all_scores, self.threshold_percentile)
            )

    def predict(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        votes = None
        all_scores = {}

        for name, head in self.heads.items():
            head.eval()
            with torch.no_grad():
                output = head(features)
                scores = output["anomaly_score"]
                all_scores[name] = scores

                threshold = self.thresholds.get(name, 0)
                vote = (scores > threshold).float()

                if votes is None:
                    votes = vote
                else:
                    votes = votes + vote

        assert votes is not None
        n_heads = len(self.heads)
        anomaly_ratio = votes / n_heads

        return {
            "anomaly_score": anomaly_ratio,
            "individual_scores": all_scores,
            "vote_counts": votes,
        }
