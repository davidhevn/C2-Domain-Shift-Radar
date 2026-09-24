import numpy as np
from typing import Dict, List, Any

class SliceLocator:
    """
    Identifies and ranks which data slices/subgroups suffer the highest domain shift or risk.
    """
    def __init__(self, detector):
        self.detector = detector

    def locate_affected_slices(
        self,
        batch_dict: Dict[str, np.ndarray],
        slice_assignments: Dict[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Args:
            batch_dict: Full batch predictions, probabilities, and features.
            slice_assignments: Dictionary mapping slice attribute name (e.g., 'class', 'camera_id')
                               to array of slice keys for each sample.
                               
        Returns:
            Ranked list of slices ordered by Shift / Risk Score.
        """
        slice_reports = []

        for attr_name, slice_keys in slice_assignments.items():
            unique_slices = np.unique(slice_keys)
            for s_val in unique_slices:
                mask = (slice_keys == s_val)
                count = int(np.sum(mask))
                if count < 10:
                    continue  # Skip slices too small to measure reliably
                
                sub_dict = {
                    k: v[mask] for k, v in batch_dict.items() if isinstance(v, np.ndarray) and len(v) == len(slice_keys)
                }
                
                score = self.detector.compute_shift_score(sub_dict)
                slice_reports.append({
                    "attribute": attr_name,
                    "slice_key": str(s_val),
                    "sample_count": count,
                    "shift_score": score
                })

        # Sort descending by shift score
        slice_reports.sort(key=lambda x: x["shift_score"], reverse=True)
        return slice_reports
