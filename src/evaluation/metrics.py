import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

@dataclass
class EvaluationSummary:
    detector_name: str
    spearman_rho: float
    spearman_pvalue: float
    shift_auroc: float
    false_alarm_count: int
    miss_count: int
    false_alarms: List[Dict[str, Any]]
    misses: List[Dict[str, Any]]

def compute_spearman_rank_correlation(shift_scores: List[float], true_drops: List[float]) -> Tuple[float, float]:
    """
    Computes Spearman rank correlation rho and p-value between shift scores and actual performance drops.
    """
    res = spearmanr(shift_scores, true_drops)
    rho = float(res.statistic) if not np.isnan(res.statistic) else 0.0
    pval = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
    return rho, pval

def compute_shift_auroc(shift_scores: List[float], is_shifted: List[int]) -> float:
    """
    Computes AUROC for classifying whether a batch is under domain shift or clean.
    """
    if len(np.unique(is_shifted)) < 2:
        return 0.5
    return float(roc_auc_score(is_shifted, shift_scores))

def analyze_alarms_and_misses(
    records: List[Dict[str, Any]],
    score_key: str,
    alert_threshold: float,
    significant_drop_threshold: float = 0.05
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Analyzes and isolates False Alarms and Misses:
    - False Alarm: Alert triggered (score >= threshold) but true drop is negligible (< significant_drop_threshold).
    - Miss: No alert triggered (score < threshold) but true drop is severe (>= significant_drop_threshold).
    """
    false_alarms = []
    misses = []

    for r in records:
        score = r[score_key]
        drop = r["true_drop"]
        alert = score >= alert_threshold
        drop_occurred = drop >= significant_drop_threshold

        item = {
            "domain": r["domain"],
            "severity": r["severity"],
            "shift_score": score,
            "true_drop": drop,
            "accuracy": r["target_acc"]
        }

        if alert and not drop_occurred:
            false_alarms.append(item)
        elif not alert and drop_occurred:
            misses.append(item)

    return false_alarms, misses
