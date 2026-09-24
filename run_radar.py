#!/usr/bin/env python3
"""
C2: Domain Shift Radar - Main Evaluation Pipeline
=================================================
Runs end-to-end evaluation:
1. Fits unsupervised detectors on Source Validation data.
2. Evaluates across multiple Target Domains (Noise, Fog, Blur, Brightness, Contrast, Pixelate)
   and multiple Severity Levels (1 to 5).
3. Computes Primary Metric: Spearman rho(ShiftScore, PerformanceDrop).
4. Computes Secondary Metrics: AUROC, Slice Localization, False Alarms & Misses.
5. Generates summary report and visualization plots.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from src.data.loader import get_cifar10_dataloaders
from src.data.corruptions import apply_domain_shift, CORRUPTIONS_REGISTRY
from src.models.model import load_model, ModelWrapper
from src.radar.detectors import (
    ATCDetector,
    EntropyDetector,
    ConfidenceDropDetector,
    MMDDetector
)
from src.radar.slice_locator import SliceLocator
from src.evaluation.metrics import (
    compute_spearman_rank_correlation,
    compute_shift_auroc,
    analyze_alarms_and_misses
)
from src.utils.visualizer import plot_spearman_correlation, plot_shift_severity_curve

def extract_batch_outputs(model: ModelWrapper, dataloader, corruption_type=None, severity=1, max_samples=1000):
    """
    Passes data through model (with optional corruption) and collects predictions and features.
    """
    model.eval()
    all_confs, all_preds, all_probs, all_targets, all_features = [], [], [], [], []
    collected = 0

    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch[0], batch[1]
            if corruption_type is not None:
                images = apply_domain_shift(images, corruption_type=corruption_type, severity=severity)

            res = model.predict_batch(images)
            all_confs.append(res["confidences"].cpu().numpy())
            all_preds.append(res["preds"].cpu().numpy())
            all_probs.append(res["probs"].cpu().numpy())
            all_features.append(res["features"].cpu().numpy())
            all_targets.append(targets.numpy())

            collected += len(images)
            if collected >= max_samples:
                break

    return {
        "confidences": np.concatenate(all_confs)[:max_samples],
        "preds": np.concatenate(all_preds)[:max_samples],
        "probs": np.concatenate(all_probs)[:max_samples],
        "features": np.concatenate(all_features)[:max_samples],
        "targets": np.concatenate(all_targets)[:max_samples],
    }

def main():
    parser = argparse.ArgumentParser(description="Run C2 Domain Shift Radar Evaluation")
    parser.add_argument("--data_dir", type=str, default="./", help="Root data directory containing cifar10")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--samples_per_eval", type=int, default=1000, help="Number of samples evaluated per test condition")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save plots and logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing C2: Domain Shift Radar Pipeline on device: {device}")

    # 1. Load Data
    print("[*] Loading CIFAR-10 data splits...")
    _, val_loader, test_loader = get_cifar10_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)

    # 2. Load Model
    print("[*] Loading Model Backbone (ResNet-18)...")
    model = load_model(device=device)

    # 3. Fit Radar Detectors on Source Validation split
    print("[*] Extracting Source Validation representations for detector calibration...")
    val_data = extract_batch_outputs(model, val_loader, max_samples=args.samples_per_eval)
    val_acc = float(np.mean(val_data["preds"] == val_data["targets"]))
    print(f"[+] Source Validation Accuracy: {val_acc * 100:.2f}%")

    detectors = {
        "ATC": ATCDetector(),
        "Entropy": EntropyDetector(),
        "ConfDrop": ConfidenceDropDetector(),
        "MMD": MMDDetector(subsample_size=300)
    }

    for name, det in detectors.items():
        det.fit(val_data, val_labels=val_data["targets"])
        print(f"    - Fitted {name} Detector.")

    # 4. Evaluate Source Clean Test
    print("[*] Evaluating Clean Source Evaluation Set...")
    clean_data = extract_batch_outputs(model, test_loader, max_samples=args.samples_per_eval)
    clean_acc = float(np.mean(clean_data["preds"] == clean_data["targets"]))
    print(f"[+] Clean Source Benchmark Accuracy: {clean_acc * 100:.2f}%")

    # 5. Evaluate Target Domains & Shift Severities
    target_domains = ["gaussian_noise", "fog", "motion_blur", "contrast", "brightness", "pixelate"]
    severities = [1, 2, 3, 4, 5]

    print(f"[*] Running multi-domain evaluation across {len(target_domains)} domains and {len(severities)} severity levels...")
    records = []

    # Include clean batch in records
    clean_rec = {
        "domain": "source_clean",
        "severity": 0,
        "is_shifted": 0,
        "target_acc": clean_acc,
        "true_drop": 0.0,
    }
    for det_name, det in detectors.items():
        clean_rec[det_name] = det.compute_shift_score(clean_data)
    records.append(clean_rec)

    for dom in target_domains:
        for sev in severities:
            tgt_data = extract_batch_outputs(
                model, test_loader, corruption_type=dom, severity=sev, max_samples=args.samples_per_eval
            )
            tgt_acc = float(np.mean(tgt_data["preds"] == tgt_data["targets"]))
            true_drop = max(0.0, clean_acc - tgt_acc)

            rec = {
                "domain": dom,
                "severity": sev,
                "is_shifted": 1,
                "target_acc": tgt_acc,
                "true_drop": true_drop,
            }

            for det_name, det in detectors.items():
                rec[det_name] = det.compute_shift_score(tgt_data)

            records.append(rec)

    # 6. Primary Metric: Spearman Rank Correlation rho
    print("\n========================================================")
    print("       C2 RADAR EVALUATION RESULTS SUMMARY              ")
    print("========================================================")
    
    true_drops = [r["true_drop"] for r in records]
    is_shifted = [r["is_shifted"] for r in records]

    results_table = []
    summary_dict = {}

    for det_name, det in detectors.items():
        scores = [r[det_name] for r in records]
        rho, pval = compute_spearman_rank_correlation(scores, true_drops)
        auroc = compute_shift_auroc(scores, is_shifted)
        
        # Analyze False Alarms & Misses
        threshold_val = float(np.percentile(scores, 50))
        fa, misses = analyze_alarms_and_misses(records, det_name, alert_threshold=threshold_val, significant_drop_threshold=0.08)

        results_table.append([
            det_name,
            f"{rho:.4f}",
            f"{pval:.2e}",
            f"{auroc:.4f}",
            len(fa),
            len(misses)
        ])
        summary_dict[det_name] = {
            "rho": rho,
            "auroc": auroc,
            "fa": fa,
            "misses": misses
        }

    headers = ["Detector", "Spearman ρ (Primary)", "p-value", "AUROC", "False Alarms", "Misses"]
    try:
        print(tabulate(results_table, headers=headers, tablefmt="github"))
    except ImportError:
        header_str = " | ".join(headers)
        print(header_str)
        print("-" * len(header_str))
        for row in results_table:
            print(" | ".join(str(c) for c in row))

    # 7. Detailed Analysis of False Alarms & Misses (Required by Challenge)
    print("\n--------------------------------------------------------")
    print("  FALSE ALARM & MISS REPORT (Production Risk Analysis)   ")
    print("--------------------------------------------------------")
    atc_summary = summary_dict["ATC"]
    if atc_summary["fa"]:
        sample_fa = atc_summary["fa"][0]
        print(f"[!] Example False Alarm: Domain '{sample_fa['domain']}' (Severity {sample_fa['severity']})")
        print(f"    Shift Score: {sample_fa['shift_score']:.4f}, but True Drop: {sample_fa['true_drop']:.4f} (Accuracy preserved at {sample_fa['accuracy']*100:.1f}%)")
    else:
        print("[i] No False Alarms observed for ATC at selected threshold.")

    if atc_summary["misses"]:
        sample_miss = atc_summary["misses"][0]
        print(f"[!] Example Miss: Domain '{sample_miss['domain']}' (Severity {sample_miss['severity']})")
        print(f"    Shift Score: {sample_miss['shift_score']:.4f}, but True Drop: {sample_miss['true_drop']:.4f} (Significant accuracy drop to {sample_miss['accuracy']*100:.1f}%)")
    else:
        print("[i] No Misses observed for ATC at selected threshold.")

    # 8. Slice Localization Demo
    print("\n--------------------------------------------------------")
    print("  SLICE LOCALIZATION DEMO (Vulnerable Slices)           ")
    print("--------------------------------------------------------")
    locator = SliceLocator(detectors["ATC"])
    severe_batch = extract_batch_outputs(model, test_loader, corruption_type="fog", severity=4, max_samples=500)
    slice_reports = locator.locate_affected_slices(severe_batch, {"class": severe_batch["targets"]})
    print("Top 3 most affected classes under Fog (Severity 4):")
    for s in slice_reports[:3]:
        print(f"  - Class ID {s['slice_key']}: Shift Score = {s['shift_score']:.4f} (Samples: {s['sample_count']})")

    # 9. Plot Results
    print("\n[*] Exporting diagnostic plots...")
    best_det = "ATC"
    plot_spearman_correlation(
        records,
        score_key=best_det,
        detector_name=best_det,
        rho=summary_dict[best_det]["rho"],
        save_path=os.path.join(args.output_dir, "spearman_correlation.png")
    )
    plot_shift_severity_curve(
        records,
        save_path=os.path.join(args.output_dir, "severity_curves.png")
    )

    print(f"\n[✔] Evaluation complete! Artifacts saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
