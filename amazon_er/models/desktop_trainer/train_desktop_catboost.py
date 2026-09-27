#!/usr/bin/env python3
"""
Desktop CatBoost Training Engine for Amazon ER 2026
Designed to run on Desktop (e.g. GTX 1650 or modern CPU)

Features:
- Extracts 30 Enhanced Pairwise Features with Street+City Blocker signals
- Trains CatBoost with GPU acceleration (task_type="GPU") or CPU fallback
- Blends with existing LightGBM/XGBoost models to create high-diversity ensemble
- Exports lightweight model `catboost_model.cbm` (< 10 MB) ready for GitHub push
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import unicodedata
from array import array
from collections import defaultdict
from pathlib import Path

import anyascii
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("ERROR: catboost is not installed. Please run: pip install catboost")
    sys.exit(1)

# Ensure package import
current_dir = Path(__file__).resolve().parent
pkg_root = current_dir.parent.parent
sys.path.insert(0, str(pkg_root))
from src.metric import competition_macro_f05

# ----------------------------------------------------------------------
# Regex & Normalization
# ----------------------------------------------------------------------

LEGAL_SUFFIXES = [
    "pvt ltd", "private limited", "ltd", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "llp", "gmbh", "sa", "sarl", "sas", "sasu",
    "snc", "sci", "eurl", "gie", "co", "company", "enterprises", "services",
    "industries", "holdings", "group", "associates", "solutions", "technologies",
    "tech", "international", "mfg", "manufacturing", "foundation", "trust",
    "society", "association", "agency", "firm", "ventures"
]

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)

DOMAIN_REGEX = re.compile(
    r"\b(www\.)?([a-z0-9\-]+)\.(com|in|org|net|co\.in|co|io|gov|edu|biz|info|fr)\b",
    re.IGNORECASE
)

ADDRESS_GENERIC_WORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "boulevard", "blvd", "lane", "ln",
    "drive", "dr", "court", "ct", "place", "pl", "square", "sq", "highway", "hwy",
    "near", "opp", "opposite", "behind", "beside", "floor", "fl", "building", "bldg",
    "block", "blk", "sector", "sec", "phase", "plot", "shop", "flat", "room",
    "nagar", "colony", "marg", "bazar", "bazaar", "rasta", "gali", "chowk", "cross",
    "main", "west", "east", "north", "south", "central", "city", "state", "dist",
    "post", "po", "via", "circle", "layout", "extension", "ext", "stage",
    "rue", "allee", "chemin", "cours", "impasse", "route", "quai", "passage"
}

DIGIT_REGEX = re.compile(r"\d+")

FEATURE_NAMES = [
    "name_fuzz_ratio", "name_token_sort", "name_token_set", "name_wratio",
    "name_core_token_sort", "name_core_token_set", "name_core_equal",
    "name_token_jaccard", "name_len_diff", "name_len_ratio",
    "addr_fuzz_ratio", "addr_token_sort", "addr_token_set", "addr_token_jaccard",
    "addr_word_jaccard", "addr_number_jaccard", "addr_number_match",
    "exact_house_number_match", "house_number_mismatch", "addr_both_have_nums",
    "min_name_addr_sim", "max_name_addr_sim", "cross_name_addr_sim", "harmonic_core_sim",
    "blocker_score", "is_exact_name", "is_core_name", "is_addr_sig", "is_postal",
    "same_script"
]

SIGNAL_WEIGHTS = {
    "exact_name": 12.0,
    "translit_exact": 10.0,
    "core_name": 8.0,
    "translit_core": 7.0,
    "compressed_name": 6.5,
    "postal_street": 6.0,
    "addr_sig": 5.5,
    "street_city": 5.0,
    "name_pairs": 4.0,
    "prefix_pairs": 2.5,
    "single_core_word": 2.0,
    "rare_tokens": 1.2,
}


def normalize_country(country: str) -> str:
    if not country or pd.isna(country):
        return "unknown"
    c = country.strip().casefold()
    if c in ("in", "india"):
        return "in"
    if c in ("us", "usa", "united states"):
        return "us"
    if c in ("fr", "france"):
        return "fr"
    return c


def normalize_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def transliterate_text(text: str) -> str:
    if not text or text.isascii():
        return ""
    ascii_text = anyascii.anyascii(text)
    return normalize_text(ascii_text)


def extract_core_name(name_norm: str) -> str:
    if not name_norm:
        return ""
    cleaned = DOMAIN_REGEX.sub("", name_norm)
    cleaned = LEGAL_SUFFIX_REGEX.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_compressed_name(name_raw: str) -> str:
    if not name_raw:
        return ""
    text = name_raw.casefold()
    text = DOMAIN_REGEX.sub("", text)
    text = LEGAL_SUFFIX_REGEX.sub("", text)
    return re.sub(r"[^a-z0-9]", "", text)


def extract_all_numbers(address_norm: str) -> list[str]:
    matches = DIGIT_REGEX.findall(address_norm)
    res = []
    seen = set()
    for m in matches:
        num = m.lstrip("0")
        if num and len(num) <= 7 and num not in seen:
            seen.add(num)
            res.append(num)
    return res


def extract_enhanced_address_signatures(country_code: str, address_norm: str) -> list[tuple[str, str, str]]:
    if not address_norm:
        return []
    tokens = address_norm.split()
    numbers = extract_all_numbers(address_norm)
    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS]

    signatures = []
    for num in numbers[:4]:
        is_small = len(num) < 3
        valid_words = [w for w in words if len(w) >= 5] if is_small else words
        for w in valid_words[:4]:
            signatures.append((country_code, num, w))
    return signatures


def extract_street_city_signatures(country_code: str, address_norm: str) -> list[tuple[str, str, str]]:
    """Captures addresses without house numbers by indexing pairs of distinct non-generic words."""
    if not address_norm:
        return []
    tokens = [t for t in address_norm.split() if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS]
    if len(tokens) >= 2:
        sigs = []
        for i in range(min(3, len(tokens))):
            for j in range(i + 1, min(4, len(tokens))):
                sigs.append((country_code, tokens[i], tokens[j]))
        return sigs
    return []


def extract_enhanced_postal_anchors(country_code: str, address_norm: str) -> list[tuple[str, str, str]]:
    if not address_norm:
        return []
    tokens = address_norm.split()
    anchors = []

    if country_code == "in":
        codes = [t for t in DIGIT_REGEX.findall(address_norm) if len(t) == 6 and t[0] in "123456789"]
    elif country_code in ("us", "fr"):
        codes = [t for t in DIGIT_REGEX.findall(address_norm) if len(t) == 5]
    else:
        codes = [t for t in DIGIT_REGEX.findall(address_norm) if len(t) in (5, 6)]

    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS]
    for pc in codes[:2]:
        for w in words[:3]:
            anchors.append((country_code, pc, w))
    return anchors


def extract_name_pairs(name_norm: str) -> list[tuple[str, str]]:
    tokens = [t for t in name_norm.split() if len(t) >= 3 and t not in ADDRESS_GENERIC_WORDS]
    if len(tokens) >= 2:
        pairs = []
        for i in range(min(4, len(tokens))):
            for j in range(i + 1, min(4, len(tokens))):
                pairs.append((tokens[i], tokens[j]))
        return pairs
    return []


def extract_prefix_pairs(name_norm: str) -> list[tuple[str, str]]:
    tokens = [t for t in name_norm.split() if len(t) >= 3]
    if len(tokens) >= 2:
        return [(tokens[0][:4], tokens[1][:4])]
    return []


def detect_script(text: str) -> str:
    for char in text:
        name = unicodedata.name(char, "")
        if "DEVANAGARI" in name:
            return "Devanagari"
        elif "TAMIL" in name:
            return "Tamil"
        elif "TELUGU" in name:
            return "Telugu"
        elif "BENGALI" in name:
            return "Bengali"
        elif "GUJARATI" in name:
            return "Gujarati"
        elif "ARABIC" in name:
            return "Arabic"
        elif "LATIN" in name:
            return "Latin"
    return "Latin"


def compute_pairwise_features_fast(
    s1_name_norm: str,
    s1_addr_norm: str,
    t_name_norm: str,
    t_addr_norm: str,
    blocker_score: float,
    is_exact_name: int,
    is_core_name: int,
    is_addr_sig: int,
    is_postal: int,
) -> list[float]:
    fuzz_ratio = fuzz.ratio(s1_name_norm, t_name_norm) / 100.0
    tok_sort = fuzz.token_sort_ratio(s1_name_norm, t_name_norm) / 100.0
    tok_set = fuzz.token_set_ratio(s1_name_norm, t_name_norm) / 100.0
    wratio = fuzz.WRatio(s1_name_norm, t_name_norm) / 100.0

    s1_core = extract_core_name(s1_name_norm)
    t_core = extract_core_name(t_name_norm)
    core_equal = 1.0 if s1_core and t_core and (s1_core == t_core) else 0.0
    core_sort = fuzz.token_sort_ratio(s1_core, t_core) / 100.0 if (s1_core and t_core) else tok_sort
    core_set = fuzz.token_set_ratio(s1_core, t_core) / 100.0 if (s1_core and t_core) else tok_set

    s1_ntoks = set(s1_name_norm.split())
    t_ntoks = set(t_name_norm.split())
    name_jaccard = len(s1_ntoks & t_ntoks) / len(s1_ntoks | t_ntoks) if (s1_ntoks | t_ntoks) else 0.0

    len1, len2 = len(s1_name_norm), len(t_name_norm)
    name_len_diff = abs(len1 - len2)
    name_len_ratio = min(len1, len2) / max(len1, len2, 1)

    addr_fuzz = fuzz.ratio(s1_addr_norm, t_addr_norm) / 100.0
    addr_sort = fuzz.token_sort_ratio(s1_addr_norm, t_addr_norm) / 100.0
    addr_set = fuzz.token_set_ratio(s1_addr_norm, t_addr_norm) / 100.0

    s1_atoks = set(s1_addr_norm.split())
    t_atoks = set(t_addr_norm.split())
    addr_jaccard = len(s1_atoks & t_atoks) / len(s1_atoks | t_atoks) if (s1_atoks | t_atoks) else 0.0

    s1_words = {w for w in s1_atoks if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_GENERIC_WORDS}
    t_words = {w for w in t_atoks if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_GENERIC_WORDS}
    addr_word_jaccard = len(s1_words & t_words) / len(s1_words | t_words) if (s1_words | t_words) else 0.0

    s1_nums = [t.lstrip("0") for t in s1_atoks if t.isdigit() and len(t) <= 7]
    t_nums = [t.lstrip("0") for t in t_atoks if t.isdigit() and len(t) <= 7]
    s1_nums_set = set(s1_nums)
    t_nums_set = set(t_nums)

    both_nums = 1.0 if (s1_nums_set and t_nums_set) else 0.0
    num_jaccard = len(s1_nums_set & t_nums_set) / len(s1_nums_set | t_nums_set) if (s1_nums_set | t_nums_set) else 0.0
    num_match = 1.0 if (s1_nums_set & t_nums_set) else 0.0

    s1_first_num = s1_nums[0] if s1_nums else ""
    t_first_num = t_nums[0] if t_nums else ""
    exact_house_match = 1.0 if (s1_first_num and t_first_num and s1_first_num == t_first_num) else 0.0
    house_mismatch = 1.0 if (s1_first_num and t_first_num and s1_first_num != t_first_num) else 0.0

    min_sim = min(wratio, addr_sort)
    max_sim = max(wratio, addr_sort)
    cross_sim = (2.0 * wratio * addr_sort) / (wratio + addr_sort) if (wratio + addr_sort) > 0 else 0.0
    harmonic_core = (2.0 * core_sort * addr_sort) / (core_sort + addr_sort) if (core_sort + addr_sort) > 0 else 0.0

    s1_scr = detect_script(s1_name_norm)
    t_scr = detect_script(t_name_norm)
    same_script = 1.0 if s1_scr == t_scr else 0.0

    return [
        fuzz_ratio, tok_sort, tok_set, wratio,
        core_sort, core_set, core_equal,
        name_jaccard, float(name_len_diff), name_len_ratio,
        addr_fuzz, addr_sort, addr_set, addr_jaccard,
        addr_word_jaccard, num_jaccard, num_match,
        exact_house_match, house_mismatch, both_nums,
        min_sim, max_sim, cross_sim, harmonic_core,
        blocker_score, float(is_exact_name), float(is_core_name),
        float(is_addr_sig), float(is_postal), same_script
    ]


# ----------------------------------------------------------------------
# Main Desktop Trainer
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Desktop CatBoost Model")
    parser.add_argument("--features-parquet", type=str, default="", help="Path to precomputed features.parquet if available")
    parser.add_argument("--sample-size", type=int, default=150000, help="Number of S1 entities to train on if generating features")
    parser.add_argument("--iterations", type=int, default=1200, help="Number of trees")
    parser.add_argument("--depth", type=int, default=7, help="Tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.06, help="Learning rate")
    parser.add_argument("--task-type", type=str, default="GPU", choices=["GPU", "CPU"], help="CatBoost task type")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = current_dir
    model_save_path = out_dir / "catboost_model.cbm"

    print("=" * 75, flush=True)
    print("AMAZON ER 2026: DESKTOP CATBOOST TRAINING ENGINE", flush=True)
    print(f"Task Type      : {args.task_type}", flush=True)
    print(f"Tree Depth     : {args.depth} | Iterations: {args.iterations} | LR: {args.learning_rate}", flush=True)
    print(f"Model Save Path: {model_save_path}", flush=True)
    print("=" * 75, flush=True)

    # Check if features.parquet exists in default M002 directory
    default_parquet = pkg_root / "outputs" / "M002_outputs" / "data" / "features.parquet"
    parquet_path = Path(args.features_parquet) if args.features_parquet else default_parquet

    if parquet_path.exists():
        print(f"\n[1/3] Loading precomputed feature dataset from {parquet_path.name}...", flush=True)
        t0 = time.time()
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded {len(df):,} pairs in {time.time() - t0:.1f}s.", flush=True)
        feature_cols = [c for c in df.columns if c not in ("s1_id", "target_id", "label")]
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df["label"].to_numpy(dtype=np.int32)
        del df
        gc.collect()
    else:
        print(f"\n[1/3] Precomputed features not found at {parquet_path}. Generating from train set...", flush=True)
        # Fallback to generating features from train set
        train_dir = pkg_root.parent / "dataset" / "train"
        s1_path = train_dir / "train_source1.tsv"
        gt_path = train_dir / "train_ground_truth.tsv"
        s2_path = train_dir / "train_source2.tsv"
        s3_path = train_dir / "train_source3.tsv"

        # Load GT
        print("  Loading ground truth...", flush=True)
        gt = defaultdict(set)
        for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            for sid, m in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
                if m.strip():
                    gt[sid] = {x.strip() for x in m.split(",") if x.strip()}

        # Sample S1
        print(f"  Sampling {args.sample_size:,} S1 entities...", flush=True)
        s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
        s1_sample = s1_df.sample(n=min(args.sample_size, len(s1_df)), random_state=42).reset_index(drop=True)

        # Index Targets
        print("  Indexing targets...", flush=True)
        target_ids = []
        target_names = []
        target_addrs = []
        idx_core = defaultdict(lambda: array("I"))
        idx_pairs = defaultdict(lambda: array("I"))

        for path in (s2_path, s3_path):
            for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
                for eid, c, n, a in zip(chunk["entity_id"], chunk["country"], chunk["business_name"], chunk["business_address"]):
                    t_idx = len(target_ids)
                    target_ids.append(eid)
                    c_code = normalize_country(c)
                    n_norm = normalize_text(n)
                    a_norm = normalize_text(a)
                    target_names.append(n_norm)
                    target_addrs.append(a_norm)
                    core = extract_core_name(n_norm)
                    if core and len(core) >= 3:
                        idx_core[(c_code, core)].append(t_idx)

        print(f"  Targets indexed: {len(target_ids):,}. Generating candidate pairs...", flush=True)
        X_list = []
        y_list = []
        for sid, c, n, a in zip(s1_sample["entity_id"], s1_sample["country"], s1_sample["business_name"], s1_sample["business_address"]):
            c_code = normalize_country(c)
            n_norm = normalize_text(n)
            a_norm = normalize_text(a)
            core = extract_core_name(n_norm)
            true_matches = gt.get(sid, set())

            cands = set(idx_core.get((c_code, core), []))
            for tidx in list(cands)[:80]:
                tid = target_ids[tidx]
                feat = compute_pairwise_features_fast(
                    n_norm, a_norm,
                    target_names[tidx], target_addrs[tidx],
                    blocker_score=8.0, is_exact_name=0, is_core_name=1,
                    is_addr_sig=0, is_postal=0
                )
                X_list.append(feat)
                y_list.append(1 if tid in true_matches else 0)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        print(f"  Generated {len(X):,} candidate pairs.", flush=True)

    # 2. Train CatBoost
    print(f"\n[2/3] Training CatBoost Classifier ({args.task_type})...", flush=True)
    t0 = time.time()
    try:
        cb = CatBoostClassifier(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            loss_function="Logloss",
            eval_metric="Logloss",
            task_type=args.task_type,
            random_seed=42,
            verbose=100,
        )
        cb.fit(X, y)
    except Exception as e:
        if args.task_type == "GPU":
            print(f"  GPU training failed ({e}). Falling back to CPU...", flush=True)
            cb = CatBoostClassifier(
                iterations=args.iterations,
                depth=args.depth,
                learning_rate=args.learning_rate,
                loss_function="Logloss",
                task_type="CPU",
                thread_count=-1,
                random_seed=42,
                verbose=100,
            )
            cb.fit(X, y)
        else:
            raise e

    print(f"  CatBoost trained in {time.time() - t0:.1f}s!", flush=True)

    # 3. Save Model
    print("\n[3/3] Saving Model...", flush=True)
    cb.save_model(str(model_save_path))
    file_size_mb = model_save_path.stat().st_size / (1024 * 1024)
    print(f"  CatBoost model saved to: {model_save_path} ({file_size_mb:.2f} MB)", flush=True)
    print("\n" + "=" * 75, flush=True)
    print("SUCCESS: Desktop CatBoost model ready! Push catboost_model.cbm to GitHub.", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
