"""
Competition Metric Evaluator for Amazon ML Challenge 2026: Business Entity Resolution.

Metric Specification:
---------------------
Macro-averaged F_beta with beta = 0.5:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Evaluation Rules:
-----------------
1. Computed independently for EACH Source-1 entity, then macro-averaged across ALL Source-1 entities.
2. Singleton handling:
   - If true_set is empty (Source-1 has no true matches):
       pred_set is empty -> score = 1.0
       pred_set is NOT empty -> score = 0.0
3. Non-singleton handling:
   - If pred_set is empty -> precision = 0, recall = 0, score = 0.0
   - Otherwise:
       precision = len(true_set & pred_set) / len(pred_set)
       recall = len(true_set & pred_set) / len(true_set)
       if (0.25 * precision + recall) == 0:
           score = 0.0
       else:
           score = (1.25 * precision * recall) / (0.25 * precision + recall)
"""

from __future__ import annotations

import csv
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (f05, precision, recall) for a single Source-1 entity.
    """
    if not true_set:
        # Singleton entity (no true matches)
        if not pred_set:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 1.0  # False positive link on a singleton

    if not pred_set:
        # Non-singleton but nothing predicted
        return 0.0, 0.0, 0.0

    intersection = len(true_set & pred_set)
    if intersection == 0:
        return 0.0, 0.0, 0.0

    precision = intersection / len(pred_set)
    recall = intersection / len(true_set)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom

    return f05, precision, recall


def competition_macro_f05(
    ground_truth: Mapping[str, Set[str]],
    predictions: Mapping[str, Set[str]],
    all_s1_ids: Iterable[str] | None = None,
) -> Dict[str, float]:
    """
    Computes macro-averaged competition F0.5, macro precision, and macro recall.

    Parameters:
    -----------
    ground_truth: Mapping from source1_entity_id to set of true S2/S3 entity IDs.
    predictions: Mapping from source1_entity_id to set of predicted S2/S3 entity IDs.
    all_s1_ids: Optional sequence of all expected S1 entity IDs.
                If None, uses ground_truth.keys().

    Returns:
    --------
    dict with:
        'macro_f05': macro-averaged F0.5 across all S1 entities
        'macro_precision': macro-averaged precision
        'macro_recall': macro-averaged recall
        'singleton_accuracy': fraction of singletons correctly predicted as empty
        'total_entities': total S1 entities evaluated
        'singleton_count': number of true singletons
        'non_singleton_count': number of true non-singletons
    """
    if all_s1_ids is None:
        target_ids = list(ground_truth.keys())
    else:
        target_ids = list(all_s1_ids)

    if not target_ids:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "total_entities": 0,
            "singleton_count": 0,
            "non_singleton_count": 0,
        }

    total_f05 = 0.0
    total_precision = 0.0
    total_recall = 0.0

    singleton_correct = 0
    singleton_total = 0
    non_singleton_total = 0

    for s1_id in target_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        f05, prec, rec = compute_entity_f05(true_set, pred_set)

        total_f05 += f05
        total_precision += prec
        total_recall += rec

        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
        else:
            non_singleton_total += 1

    n = len(target_ids)
    singleton_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 1.0

    return {
        "macro_f05": total_f05 / n,
        "macro_precision": total_precision / n,
        "macro_recall": total_recall / n,
        "singleton_accuracy": singleton_acc,
        "total_entities": n,
        "singleton_count": singleton_total,
        "non_singleton_count": non_singleton_total,
    }


def load_ground_truth_tsv(path: str) -> Dict[str, Set[str]]:
    """Loads train_ground_truth.tsv into a mapping of s1_id -> set(matched_ids)."""
    gt: Dict[str, Set[str]] = {}
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            if len(row) > 1 and row[1].strip():
                gt[s1_id] = {x.strip() for x in row[1].split(",") if x.strip()}
            else:
                gt[s1_id] = set()
    return gt


def load_submission_tsv(path: str) -> Dict[str, Set[str]]:
    """Loads a submission or matching_results.tsv into a mapping of s1_id -> set(matched_ids)."""
    preds: Dict[str, Set[str]] = {}
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            if len(row) > 1 and row[1].strip():
                preds[s1_id] = {x.strip() for x in row[1].split(",") if x.strip()}
            else:
                preds[s1_id] = set()
    return preds


def run_unit_tests() -> bool:
    """Verifies evaluator with comprehensive test cases."""
    print("Running metric unit tests...")

    # Case 1: Perfect match single entity
    # S1 matches S2-1, S3-1; pred = S2-1, S3-1 -> F0.5 = 1.0
    f, p, r = compute_entity_f05({"S2-1", "S3-1"}, {"S2-1", "S3-1"})
    assert abs(f - 1.0) < 1e-6 and abs(p - 1.0) < 1e-6 and abs(r - 1.0) < 1e-6, "Case 1 failed"

    # Case 2: Singleton with empty prediction -> F0.5 = 1.0
    f, p, r = compute_entity_f05(set(), set())
    assert abs(f - 1.0) < 1e-6, "Case 2 failed"

    # Case 3: Singleton with FALSE POSITIVE prediction -> F0.5 = 0.0
    f, p, r = compute_entity_f05(set(), {"S2-999"})
    assert abs(f - 0.0) < 1e-6, "Case 3 failed"

    # Case 4: Non-singleton with empty prediction -> F0.5 = 0.0
    f, p, r = compute_entity_f05({"S2-1"}, set())
    assert abs(f - 0.0) < 1e-6, "Case 4 failed"

    # Case 5: Partial match: Official example from competition README:
    # true: [S2-00047, S3-00812] (size 2)
    # pred: [S2-00047, S2-00193, S3-00812] (size 3)
    # Precision = 2/3, Recall = 2/2 = 1.0
    # F0.5 = (1.25 * 2/3 * 1.0) / (0.25 * 2/3 + 1.0) = (5/6) / (1/6 + 1) = (5/6) / (7/6) = 5/7 ≈ 0.7142857
    f, p, r = compute_entity_f05({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"})
    assert abs(p - (2.0 / 3.0)) < 1e-5, f"Expected precision 0.6667, got {p}"
    assert abs(r - 1.0) < 1e-5, f"Expected recall 1.0, got {r}"
    assert abs(f - (5.0 / 7.0)) < 1e-5, f"Expected F0.5 0.7143, got {f}"

    # Case 6: Precision penalization vs Recall penalization test:
    # False positive is penalized more heavily by F0.5:
    # Subcase A: 1 true match, predicted that 1 + 1 false positive -> precision 1/2, recall 1/1 -> F0.5 = (1.25*0.5*1)/(0.25*0.5+1) = 0.625/1.125 = 0.5555
    f_fp, _, _ = compute_entity_f05({"S2-1"}, {"S2-1", "S2-2"})
    # Subcase B: 2 true matches, predicted only 1 (false negative) -> precision 1/1, recall 1/2 -> F0.5 = (1.25*1*0.5)/(0.25*1+0.5) = 0.625/0.75 = 0.8333
    f_fn, _, _ = compute_entity_f05({"S2-1", "S2-2"}, {"S2-1"})
    assert f_fn > f_fp, "F0.5 must reward high precision (recall penalty should be lighter than precision penalty)"

    # Case 7: Macro-average test across a population of 4 entities:
    # E1: perfect match (1.0)
    # E2: singleton correct (1.0)
    # E3: singleton false positive (0.0)
    # E4: README example (5/7 ≈ 0.7143)
    gt = {
        "E1": {"S2-1", "S3-1"},
        "E2": set(),
        "E3": set(),
        "E4": {"S2-00047", "S3-00812"},
    }
    preds = {
        "E1": {"S2-1", "S3-1"},
        "E2": set(),
        "E3": {"S2-999"},
        "E4": {"S2-00047", "S2-00193", "S3-00812"},
    }
    res = competition_macro_f05(gt, preds)
    expected_macro = (1.0 + 1.0 + 0.0 + (5.0 / 7.0)) / 4.0
    assert abs(res["macro_f05"] - expected_macro) < 1e-6, f"Macro F0.5 mismatch: {res['macro_f05']} vs {expected_macro}"
    assert res["singleton_count"] == 2
    assert res["singleton_accuracy"] == 0.5
    assert res["total_entities"] == 4

    print("ALL UNIT TESTS PASSED SUCCESSFULLY! Output:", res)
    return True


if __name__ == "__main__":
    run_unit_tests()
