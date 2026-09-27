#!/usr/bin/env python3
"""
Train XGBoost Matcher on M002 Features (8.8M candidate pairs)
Saves model to amazon_er/outputs/M002_outputs/models/xgb_model.json
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

def main():
    base_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource")
    m002_data_dir = base_dir / "student_resource" / "amazon_er" / "outputs" / "M002_outputs" / "data"
    m002_models_dir = base_dir / "student_resource" / "amazon_er" / "outputs" / "M002_outputs" / "models"
    m002_models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAINING XGBOOST MATCHER ON M002 FEATURE DATASET (8.8M PAIRS)")
    print("=" * 70)

    # 1. Load Features
    print("\n[1/3] Loading features.parquet...")
    t0 = time.time()
    df = pd.read_parquet(m002_data_dir / "features.parquet")
    print(f"  Loaded {len(df):,} rows in {time.time() - t0:.1f}s.")

    feature_cols = [c for c in df.columns if c not in ("s1_id", "target_id", "label")]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    del df  # free memory

    # 2. Train XGBoost with histogram method
    print("\n[2/3] Training XGBoost (tree_method='hist', max_depth=6, 300 trees)...")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    print(f"  XGBoost trained in {time.time() - t0:.1f}s.")

    # 3. Save Model
    out_path = m002_models_dir / "xgb_model.json"
    model.save_model(str(out_path))
    print(f"\n[3/3] Model saved successfully to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
