#!/usr/bin/env python3
"""
E006: Hybrid 4-Step Resolution Pipeline Benchmark on 1,000 Validation Entities

Steps Evaluated:
1. Fast Sparse Candidate Generation (Fixed from M002)
2. Tabular GBDT Matcher: LightGBM vs XGBoost vs Blend (LGBM + XGB)
3. Selective Cross-Encoder Reranker: mmarco-mMiniLMv2-L12-H384-v1 on borderline pairs (p in [0.42, 0.68])
4. Bipartite Disambiguation (1-to-many mutual exclusivity)

Run command:
  C:\\Users\\Shardul\\AppData\\Local\\Python\\bin\\python.exe amazon_er/experiments/E006_hybrid_test/test_hybrid_1000.py --sample-size 1000
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
import xgboost as xgb

# Ensure package import
current_dir = Path(__file__).resolve().parent
pkg_root = current_dir.parent.parent
sys.path.insert(0, str(pkg_root))
from src.metric import competition_macro_f05


def main():
    parser = argparse.ArgumentParser(description="Test Hybrid Pipeline on Validation Slice")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of S1 validation entities (default: 1000)")
    parser.add_argument("--model-name", type=str, default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", help="CrossEncoder model")
    args = parser.parse_args()

    start_time = time.time()
    base_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource")
    m002_data_dir = base_dir / "student_resource" / "amazon_er" / "outputs" / "M002_outputs" / "data"
    m002_models_dir = base_dir / "student_resource" / "amazon_er" / "outputs" / "M002_outputs" / "models"
    gt_path = base_dir / "dataset" / "train" / "train_ground_truth.tsv"
    s1_path = base_dir / "dataset" / "train" / "train_source1.tsv"
    s2_path = base_dir / "dataset" / "train" / "train_source2.tsv"
    s3_path = base_dir / "dataset" / "train" / "train_source3.tsv"

    out_dir = base_dir / "student_resource" / "amazon_er" / "outputs" / "E006_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"E006: HYBRID PIPELINE BENCHMARK ({args.sample_size:,} VALIDATION ENTITIES)")
    print(f"Cross-Encoder Model: {args.model_name}")
    print("=" * 75)

    # 1. Load M002 Features & OOF Predictions
    print("\n[1/5] Loading M002 OOF predictions and candidate metadata...")
    oof_df = pd.read_parquet(m002_data_dir / "oof_predictions.parquet")
    features_df = pd.read_parquet(m002_data_dir / "features.parquet")

    # Select random slice of N S1 entities from M002
    np.random.seed(42)
    all_s1 = sorted(list(set(oof_df["s1_id"])))
    selected_s1 = set(np.random.choice(all_s1, size=min(args.sample_size, len(all_s1)), replace=False))
    selected_s1_list = sorted(list(selected_s1))
    print(f"  Selected {len(selected_s1_list):,} S1 entities for benchmark.")

    # Filter slice
    mask = oof_df["s1_id"].isin(selected_s1)
    slice_oof = oof_df[mask].copy().reset_index(drop=True)
    slice_features = features_df[mask].copy().reset_index(drop=True)
    print(f"  Slice contains {len(slice_oof):,} candidate pairs (mean {len(slice_oof)/len(selected_s1_list):.1f} cands/entity).")

    # Load Ground Truth for this slice
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    matched_gt = gt_df[gt_df["source1_entity_id"].isin(selected_s1)]
    gt_dict = {}
    for sid, m in zip(matched_gt["source1_entity_id"].astype(str), matched_gt["matched_entity_ids"].astype(str)):
        m_str = m.strip()
        gt_dict[sid] = {x.strip() for x in m_str.split(",") if x.strip()} if m_str else set()

    for sid in selected_s1_list:
        if sid not in gt_dict:
            gt_dict[sid] = set()

    total_true = sum(len(v) for v in gt_dict.values())
    print(f"  Ground truth loaded: {total_true:,} true matches across {len(selected_s1_list):,} entities.")

    # 2. Baseline A: Pure LightGBM from M002
    print("\n[2/5] Evaluating Baseline LightGBM on slice...")
    p_lgb = slice_oof["proba"].to_numpy(dtype=np.float32)

    # Helper for evaluation
    def evaluate_predictions(s1_ids, target_ids, probas, thresh, mode="bipartite"):
        pred_dict = defaultdict(set)
        if mode == "bipartite":
            target_claimed = {}
            for sid, tid, p in zip(s1_ids, target_ids, probas):
                if p >= thresh:
                    if tid not in target_claimed or p > target_claimed[tid][0]:
                        target_claimed[tid] = (p, sid)
            for tid, (p, sid) in target_claimed.items():
                pred_dict[sid].add(tid)
        else:
            for sid, tid, p in zip(s1_ids, target_ids, probas):
                if p >= thresh:
                    pred_dict[sid].add(tid)

        return competition_macro_f05(gt_dict, pred_dict, all_s1_ids=selected_s1_list)

    s1_col = slice_oof["s1_id"].astype(str).tolist()
    target_col = slice_oof["target_id"].astype(str).tolist()

    res_lgb = evaluate_predictions(s1_col, target_col, p_lgb, thresh=0.62, mode="bipartite")
    print(f"  Pure LightGBM (thresh=0.62, Bipartite): Macro-F0.5 = {res_lgb['macro_f05']:.4f} (Prec: {res_lgb['macro_precision']:.4f}, Rec: {res_lgb['macro_recall']:.4f})")

    # 3. Step 2 Blend: LightGBM + XGBoost
    print("\n[3/5] Training fast XGBoost model on features to evaluate GBDT architectural blend...")
    feature_cols = [c for c in slice_features.columns if c not in ("s1_id", "target_id", "label")]
    X_mat = slice_features[feature_cols].to_numpy(dtype=np.float32)
    y_vec = slice_features["label"].to_numpy(dtype=np.int32)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    t_xgb = time.time()
    xgb_model.fit(X_mat, y_vec)
    p_xgb = xgb_model.predict_proba(X_mat)[:, 1]
    print(f"  XGBoost trained in {time.time() - t_xgb:.1f}s.")

    # Blend: 50% LGBM + 50% XGB
    p_blend = 0.5 * p_lgb + 0.5 * p_xgb
    res_blend = evaluate_predictions(s1_col, target_col, p_blend, thresh=0.60, mode="bipartite")
    print(f"  LightGBM + XGBoost Blend (thresh=0.60, Bipartite): Macro-F0.5 = {res_blend['macro_f05']:.4f} (Prec: {res_blend['macro_precision']:.4f}, Rec: {res_blend['macro_recall']:.4f})")

    # 4. Step 3: Selective Cross-Encoder Reranking
    print(f"\n[4/5] Running Selective Cross-Encoder ({args.model_name}) on borderline pairs...")
    # Find borderline pairs: p between 0.40 and 0.68
    borderline_mask = (p_blend >= 0.40) & (p_blend <= 0.68)
    n_borderline = int(borderline_mask.sum())
    print(f"  Found {n_borderline:,} borderline pairs ({n_borderline/len(slice_oof)*100:.1f}% of candidate pairs).")

    if n_borderline > 0:
        # Load string text lookups for borderline pairs
        needed_s1 = set(slice_oof.loc[borderline_mask, "s1_id"])
        needed_target = set(slice_oof.loc[borderline_mask, "target_id"])

        s1_strings = {}
        for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_s1)]
            for eid, name, addr in zip(m["entity_id"].astype(str), m["business_name"].astype(str), m["business_address"].astype(str)):
                s1_strings[eid] = f"{name} | {addr}"

        target_strings = {}
        for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_target)]
            for eid, name, addr in zip(m["entity_id"].astype(str), m["business_name"].astype(str), m["business_address"].astype(str)):
                target_strings[eid] = f"{name} | {addr}"

        for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_target)]
            for eid, name, addr in zip(m["entity_id"].astype(str), m["business_name"].astype(str), m["business_address"].astype(str)):
                target_strings[eid] = f"{name} | {addr}"

        # Build pair text inputs
        borderline_indices = np.where(borderline_mask)[0]
        ce_pairs = []
        for idx in borderline_indices:
            sid = s1_col[idx]
            tid = target_col[idx]
            txt1 = s1_strings.get(sid, "")
            txt2 = target_strings.get(tid, "")
            ce_pairs.append([txt1, txt2])

        print(f"  Loading CrossEncoder model '{args.model_name}'...")
        t_ce_load = time.time()
        ce_model = CrossEncoder(args.model_name, max_length=256)
        print(f"  Model loaded in {time.time() - t_ce_load:.1f}s.")

        print(f"  Scoring {len(ce_pairs):,} borderline pairs...")
        t_ce_infer = time.time()
        raw_ce_scores = ce_model.predict(ce_pairs, batch_size=64, show_progress_bar=True)
        ce_infer_time = time.time() - t_ce_infer
        print(f"  Cross-Encoder scored {len(ce_pairs):,} pairs in {ce_infer_time:.2f}s ({len(ce_pairs)/ce_infer_time:.1f} pairs/sec)!")

        # Apply sigmoid if raw logits
        if raw_ce_scores.min() < 0.0 or raw_ce_scores.max() > 1.0:
            ce_probas = 1.0 / (1.0 + np.exp(-raw_ce_scores))
        else:
            ce_probas = raw_ce_scores

        # Update hybrid probabilities: 60% GBDT + 40% Cross-Encoder for borderlines
        p_hybrid = p_blend.copy()
        for idx, ce_p in zip(borderline_indices, ce_probas):
            p_hybrid[idx] = 0.55 * p_blend[idx] + 0.45 * float(ce_p)

        res_hybrid = evaluate_predictions(s1_col, target_col, p_hybrid, thresh=0.60, mode="bipartite")
        print(f"  Hybrid (LGBM + XGB + CrossEncoder, Bipartite): Macro-F0.5 = {res_hybrid['macro_f05']:.4f} (Prec: {res_hybrid['macro_precision']:.4f}, Rec: {res_hybrid['macro_recall']:.4f})")
    else:
        res_hybrid = res_blend

    # 5. Final Summary Table
    print("\n" + "=" * 75)
    print("HYBRID PIPELINE BENCHMARK RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Pipeline Configuration':<45} | {'Macro-F0.5':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 75)
    print(f"{'1. Pure LightGBM (M002)':<45} | {res_lgb['macro_f05']:<10.4f} | {res_lgb['macro_precision']:<10.4f} | {res_lgb['macro_recall']:<10.4f}")
    print(f"{'2. LightGBM + XGBoost Blend':<45} | {res_blend['macro_f05']:<10.4f} | {res_blend['macro_precision']:<10.4f} | {res_blend['macro_recall']:<10.4f}")
    print(f"{'3. Hybrid (GBDT + CrossEncoder + Bipartite)':<45} | {res_hybrid['macro_f05']:<10.4f} | {res_hybrid['macro_precision']:<10.4f} | {res_hybrid['macro_recall']:<10.4f}")
    print("=" * 75)

    results_payload = {
        "sample_size": args.sample_size,
        "pure_lgbm_f05": res_lgb["macro_f05"],
        "blend_gbdt_f05": res_blend["macro_f05"],
        "hybrid_ce_f05": res_hybrid["macro_f05"],
        "total_runtime_seconds": time.time() - start_time,
    }
    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"Results saved to {out_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
