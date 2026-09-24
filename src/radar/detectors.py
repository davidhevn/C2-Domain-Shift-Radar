import abc
import numpy as np
import torch
from typing import Dict, Any, Optional
from sklearn.metrics.pairwise import rbf_kernel

class BaseShiftDetector(abc.ABC):
    """
    Abstract base class for unsupervised domain shift & risk signal detectors.
    """
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abc.abstractmethod
    def fit(self, val_dict: Dict[str, np.ndarray], val_labels: Optional[np.ndarray] = None) -> None:
        """
        Calibrate detector using source validation distribution (with ground-truth labels if available).
        """
        pass

    @abc.abstractmethod
    def compute_shift_score(self, batch_dict: Dict[str, np.ndarray]) -> float:
        """
        Compute scalar shift / risk score on an unlabeled target batch.
        Higher score must indicate higher probability of performance degradation.
        """
        pass

class ATCDetector(BaseShiftDetector):
    """
    Average Thresholded Confidence (NeurIPS 2021).
    Finds threshold t on validation set matching validation accuracy.
    Estimates accuracy on target as fraction of instances with confidence >= t.
    Shift score = Estimated Performance Drop (Val Acc - Target Est Acc).
    """
    def __init__(self):
        super().__init__(name="ATC (Average Thresholded Confidence)")
        self.threshold = 0.5
        self.val_acc = 1.0

    def fit(self, val_dict: Dict[str, np.ndarray], val_labels: Optional[np.ndarray] = None) -> None:
        confs = val_dict["confidences"]
        preds = val_dict["preds"]
        
        if val_labels is not None:
            self.val_acc = float(np.mean(preds == val_labels))
        else:
            self.val_acc = float(np.mean(confs))

        # Find threshold t such that mean(confs >= t) ~= val_acc
        sorted_confs = np.sort(confs)
        n = len(sorted_confs)
        idx = max(0, min(n - 1, int(np.round((1.0 - self.val_acc) * n))))
        self.threshold = float(sorted_confs[idx])
        self.is_fitted = True

    def compute_shift_score(self, batch_dict: Dict[str, np.ndarray]) -> float:
        confs = batch_dict["confidences"]
        estimated_acc = float(np.mean(confs >= self.threshold))
        estimated_drop = max(0.0, self.val_acc - estimated_acc)
        return estimated_drop

class EntropyDetector(BaseShiftDetector):
    """
    Uncertainty detector based on predictive Shannon entropy.
    Shift score = Mean Target Entropy - Mean Baseline Val Entropy.
    """
    def __init__(self):
        super().__init__(name="Softmax Entropy (Uncertainty)")
        self.baseline_entropy = 0.0

    def fit(self, val_dict: Dict[str, np.ndarray], val_labels: Optional[np.ndarray] = None) -> None:
        probs = val_dict["probs"]
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        self.baseline_entropy = float(np.mean(entropy))
        self.is_fitted = True

    def compute_shift_score(self, batch_dict: Dict[str, np.ndarray]) -> float:
        probs = batch_dict["probs"]
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        current_entropy = float(np.mean(entropy))
        return max(0.0, current_entropy - self.baseline_entropy)

class ConfidenceDropDetector(BaseShiftDetector):
    """
    Confidence drop detector (NannyML CBPE core intuition).
    Shift score = Baseline Mean Confidence - Current Mean Confidence.
    """
    def __init__(self):
        super().__init__(name="Max-Confidence Drop")
        self.baseline_conf = 1.0

    def fit(self, val_dict: Dict[str, np.ndarray], val_labels: Optional[np.ndarray] = None) -> None:
        self.baseline_conf = float(np.mean(val_dict["confidences"]))
        self.is_fitted = True

    def compute_shift_score(self, batch_dict: Dict[str, np.ndarray]) -> float:
        current_conf = float(np.mean(batch_dict["confidences"]))
        return max(0.0, self.baseline_conf - current_conf)

class MMDDetector(BaseShiftDetector):
    """
    Maximum Mean Discrepancy (MMD) on latent feature embeddings using RBF kernel.
    """
    def __init__(self, subsample_size: int = 500, gamma: Optional[float] = None):
        super().__init__(name="MMD (Feature Representation Drift)")
        self.subsample_size = subsample_size
        self.gamma = gamma
        self.val_features = None

    def fit(self, val_dict: Dict[str, np.ndarray], val_labels: Optional[np.ndarray] = None) -> None:
        feats = val_dict["features"]
        if len(feats) > self.subsample_size:
            idx = np.random.choice(len(feats), self.subsample_size, replace=False)
            self.val_features = feats[idx]
        else:
            self.val_features = feats
        self.is_fitted = True

    def compute_shift_score(self, batch_dict: Dict[str, np.ndarray]) -> float:
        tgt_feats = batch_dict["features"]
        if len(tgt_feats) > self.subsample_size:
            idx = np.random.choice(len(tgt_feats), self.subsample_size, replace=False)
            tgt_feats = tgt_feats[idx]

        src = self.val_features
        # Compute MMD^2 with RBF kernel
        gamma = self.gamma if self.gamma is not None else (1.0 / src.shape[1])
        k_xx = rbf_kernel(src, src, gamma=gamma)
        k_yy = rbf_kernel(tgt_feats, tgt_feats, gamma=gamma)
        k_xy = rbf_kernel(src, tgt_feats, gamma=gamma)

        mmd_sq = np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy)
        return float(max(0.0, mmd_sq))

class RadarEnsemble:
    """
    Ensemble aggregator running multiple detectors simultaneously.
    """
    def __init__(self):
        self.detectors = [
            ATCDetector(),
            EntropyDetector(),
            ConfidenceDropDetector(),
            MMDDetector()
        ]

    def fit(self, val_dict: Dict[str, np.ndarray], val_labels: Optional[np.ndarray] = None):
        for d in self.detectors:
            d.fit(val_dict, val_labels)

    def evaluate_batch(self, batch_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        results = {}
        for d in self.detectors:
            results[d.name] = d.compute_shift_score(batch_dict)
        return results
