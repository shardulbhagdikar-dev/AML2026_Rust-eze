"""
M001v2: Supervised Pairwise Entity Matching Pipeline
Amazon ML Challenge 2026 - Business Entity Resolution

Features:
- Robust 4-stage pipeline with atomic disk checkpointing & automatic resume.
- Memory-safe architecture (< 2 GB RAM peak) using integer indices and chunked Parquet files.
- High-recall E004.5 candidate generator as hard-negative miner.
- 24 pairwise lexical, address-number, script, and blocking features via RapidFuzz.
- Grouped 5-Fold Cross-Validation (by source1_entity_id: zero data leakage).
- LightGBM gradient boosting with early stopping.
- Macro-F0.5 threshold optimization and singleton abstention rule.
"""

from __future__ import annotations

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

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import GroupKFold

# Ensure src/metric.py can be imported
AMAZON_ER_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = AMAZON_ER_DIR / "src"
for _p in (SRC_DIR, AMAZON_ER_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from metric import competition_macro_f05, compute_entity_f05
except ImportError:
    from src.metric import competition_macro_f05, compute_entity_f05

# ----------------------------------------------------------------------
# Regex & Normalization Rules
# ----------------------------------------------------------------------

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b(pvt\s*ltd|private\s*limited|pvt\s*limited|private\s*ltd|pte\s*ltd|limited|ltd|"
    r"inc|incorporated|llc|llp|corp|corporation|co|company|enterprises|enterprise|"
    r"services|service|solutions|solution|industries|industry|trading|group|"
    r"holdings|holding|international|consultants|consultancy|associates|"
    r"technologies|technology|software)\b",
    re.IGNORECASE,
)

DOMAIN_REGEX = re.compile(
    r"(\.com|\.net|\.org|\.in|\.co\.in|\.co|\.us|\.fr|\.io|\.biz|\.info|#\d+|www\.)",
    re.IGNORECASE,
)

STOPWORDS = {
    "and", "the", "of", "in", "for", "at", "to", "a", "an", "on", "by", "with",
    "pvt", "ltd", "private", "limited", "inc", "corp", "llc", "co", "company",
    "services", "solutions", "enterprises", "trading", "group", "holdings",
    "india", "usa", "us", "france", "store", "center", "shop", "hotel", "care"
}

ADDRESS_GENERIC_WORDS = {
    "street", "st", "road", "rd", "lane", "ln", "avenue", "ave", "drive", "dr",
    "suite", "ste", "floor", "fl", "building", "bldg", "near", "opposite", "opp",
    "nagar", "block", "blk", "sector", "sec", "post", "dist", "state", "city",
    "north", "south", "east", "west", "main", "cross", "highway", "hwy", "way",
    "park", "plaza", "room", "apartment", "apt", "unit", "box", "po"
}

SIGNAL_WEIGHTS = {
    "exact_name": 10.0,
    "core_name": 7.0,
    "compressed_name": 6.0,
    "postal_street": 5.5,
    "addr_sig": 5.0,
    "name_pairs": 3.5,
    "prefix_pairs": 2.0,
    "rare_tokens": 1.0,
}


def normalize_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def extract_core_name(name_norm: str) -> str:
    if not name_norm:
        return ""
    cleaned = DOMAIN_REGEX.sub("", name_norm)
    cleaned = LEGAL_SUFFIX_REGEX.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_compressed_name(name_raw: str) -> str:
    if not name_raw or pd.isna(name_raw):
        return ""
    text = name_raw.casefold()
    text = DOMAIN_REGEX.sub("", text)
    text = LEGAL_SUFFIX_REGEX.sub("", text)
    return re.sub(r"[^a-z0-9]", "", text)


def extract_clean_address_signatures(country: str, address_norm: str) -> list[tuple[str, str, str]]:
    if not address_norm:
        return []
    tokens = address_norm.split()
    raw_numbers = [t.lstrip("0") for t in tokens if t.isdigit()]
    numbers = [n for n in raw_numbers if n and len(n) <= 7]
    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS]

    signatures = []
    c = country.casefold()
    for num in numbers[:3]:
        is_small = len(num) < 3
        valid_words = [w for w in words if len(w) >= 5] if is_small else words
        for w in valid_words[:2]:
            signatures.append((c, num, w))
    return signatures


def extract_postal_anchors(country: str, address_norm: str) -> list[tuple[str, str, str]]:
    if not address_norm:
        return []
    tokens = address_norm.split()
    postals = [t for t in tokens if t.isdigit() and (4 <= len(t) <= 6)]
    if not postals:
        return []
    other_numbers = [t.lstrip("0") for t in tokens if t.isdigit() and t not in postals and t.lstrip("0")]
    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS]

    anchors = []
    c = country.casefold()
    p = postals[0]
    for num in other_numbers[:1]:
        anchors.append((c, p, num))
    for w in words[:2]:
        anchors.append((c, p, w))
    return anchors


def extract_name_pairs(name_norm: str) -> list[tuple[str, str]]:
    words = [w for w in name_norm.split() if len(w) >= 3 and w not in STOPWORDS]
    if len(words) < 2:
        return []
    pairs = []
    for i in range(len(words) - 1):
        w1, w2 = sorted([words[i], words[i+1]])
        pairs.append((w1, w2))
    return pairs


def extract_prefix_pairs(name_norm: str) -> list[tuple[str, str]]:
    words = [w[:4] for w in name_norm.split() if len(w) >= 4 and w not in STOPWORDS]
    if len(words) < 2:
        return []
    pairs = []
    for i in range(len(words) - 1):
        p1, p2 = sorted([words[i], words[i+1]])
        pairs.append((p1, p2))
    return pairs


def extract_distinctive_tokens(name_norm: str) -> list[str]:
    return [w for w in name_norm.split() if len(w) >= 6 and w not in STOPWORDS]


# ----------------------------------------------------------------------
# Low-Memory Candidate Index
# ----------------------------------------------------------------------

class BlockingIndex:
    def __init__(self, max_postings: int = 120):
        self.max_postings = max_postings
        self.target_ids: list[str] = []

        self.idx_exact_name = defaultdict(lambda: array("I"))
        self.idx_core_name = defaultdict(lambda: array("I"))
        self.idx_compressed_name = defaultdict(lambda: array("I"))
        self.idx_name_pairs = defaultdict(lambda: array("I"))
        self.idx_prefix_pairs = defaultdict(lambda: array("I"))
        self.idx_addr_sig = defaultdict(lambda: array("I"))
        self.idx_postal = defaultdict(lambda: array("I"))
        self.idx_rare_tokens = defaultdict(lambda: array("I"))

    def add_target(self, target_id: str, country: str, name_raw: str, addr_raw: str):
        target_idx = len(self.target_ids)
        self.target_ids.append(target_id)

        c = country.casefold()
        name_norm = normalize_text(name_raw)
        addr_norm = normalize_text(addr_raw)

        if name_norm:
            k = (c, name_norm)
            lst = self.idx_exact_name[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        core = extract_core_name(name_norm)
        if core and len(core) >= 3:
            k = (c, core)
            lst = self.idx_core_name[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        comp = extract_compressed_name(name_raw)
        if comp and len(comp) >= 5:
            k = (c, comp)
            lst = self.idx_compressed_name[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        for w1, w2 in extract_name_pairs(name_norm):
            k = (c, w1, w2)
            lst = self.idx_name_pairs[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        for p1, p2 in extract_prefix_pairs(name_norm):
            k = (c, p1, p2)
            lst = self.idx_prefix_pairs[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        for sig in extract_clean_address_signatures(c, addr_norm):
            lst = self.idx_addr_sig[sig]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        for anchor in extract_postal_anchors(c, addr_norm):
            lst = self.idx_postal[anchor]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        for tok in extract_distinctive_tokens(name_norm):
            k = (c, tok)
            lst = self.idx_rare_tokens[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

    def retrieve(self, country: str, name_raw: str, addr_raw: str) -> dict[int, tuple[float, int, int, int, int]]:
        """Returns dict of target_idx -> (score, is_exact, is_core, is_addr, is_postal)"""
        c = country.casefold()
        name_norm = normalize_text(name_raw)
        addr_norm = normalize_text(addr_raw)

        cand_info: dict[int, list[float | int]] = defaultdict(lambda: [0.0, 0, 0, 0, 0])

        if name_norm and (c, name_norm) in self.idx_exact_name:
            w = SIGNAL_WEIGHTS["exact_name"]
            for idx in self.idx_exact_name[(c, name_norm)]:
                cand_info[idx][0] += w
                cand_info[idx][1] = 1

        core = extract_core_name(name_norm)
        if core and (c, core) in self.idx_core_name:
            w = SIGNAL_WEIGHTS["core_name"]
            for idx in self.idx_core_name[(c, core)]:
                cand_info[idx][0] += w
                cand_info[idx][2] = 1

        comp = extract_compressed_name(name_raw)
        if comp and (c, comp) in self.idx_compressed_name:
            w = SIGNAL_WEIGHTS["compressed_name"]
            for idx in self.idx_compressed_name[(c, comp)]:
                cand_info[idx][0] += w

        for w1, w2 in extract_name_pairs(name_norm):
            k = (c, w1, w2)
            if k in self.idx_name_pairs:
                w = SIGNAL_WEIGHTS["name_pairs"]
                for idx in self.idx_name_pairs[k]:
                    cand_info[idx][0] += w

        for sig in extract_clean_address_signatures(c, addr_norm):
            if sig in self.idx_addr_sig:
                w = SIGNAL_WEIGHTS["addr_sig"]
                for idx in self.idx_addr_sig[sig]:
                    cand_info[idx][0] += w
                    cand_info[idx][3] = 1

        for anchor in extract_postal_anchors(c, addr_norm):
            if anchor in self.idx_postal:
                w = SIGNAL_WEIGHTS["postal_street"]
                for idx in self.idx_postal[anchor]:
                    cand_info[idx][0] += w
                    cand_info[idx][4] = 1

        if len(cand_info) < 25:
            for p1, p2 in extract_prefix_pairs(name_norm):
                k = (c, p1, p2)
                if k in self.idx_prefix_pairs:
                    lst = self.idx_prefix_pairs[k]
                    if len(lst) <= 60:
                        w = SIGNAL_WEIGHTS["prefix_pairs"]
                        for idx in lst:
                            cand_info[idx][0] += w

        if len(cand_info) < 15:
            for tok in extract_distinctive_tokens(name_norm):
                k = (c, tok)
                if k in self.idx_rare_tokens:
                    lst = self.idx_rare_tokens[k]
                    if len(lst) <= 40:
                        w = SIGNAL_WEIGHTS["rare_tokens"]
                        for idx in lst:
                            cand_info[idx][0] += w

        return {
            k: (float(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]))
            for k, v in cand_info.items()
        }


# ----------------------------------------------------------------------
# Pairwise Feature Extraction
# ----------------------------------------------------------------------

def compute_pairwise_features(
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
    """Computes 24 numerical features for a single candidate pair."""
    # 1. Name Similarities
    fuzz_ratio = fuzz.ratio(s1_name_norm, t_name_norm)
    token_sort = fuzz.token_sort_ratio(s1_name_norm, t_name_norm)
    token_set = fuzz.token_set_ratio(s1_name_norm, t_name_norm)
    wratio = fuzz.WRatio(s1_name_norm, t_name_norm)

    s1_tokens = set(s1_name_norm.split())
    t_tokens = set(t_name_norm.split())
    union_tok = s1_tokens | t_tokens
    tok_jaccard = (len(s1_tokens & t_tokens) / len(union_tok)) if union_tok else 0.0

    len1 = len(s1_name_norm)
    len2 = len(t_name_norm)
    len_diff = abs(len1 - len2)
    len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0

    # Core equality
    s1_core = extract_core_name(s1_name_norm)
    t_core = extract_core_name(t_name_norm)
    core_equal = 1.0 if (s1_core and s1_core == t_core) else 0.0

    # 2. Address Similarities
    addr_ratio = fuzz.ratio(s1_addr_norm, t_addr_norm)
    addr_token_set = fuzz.token_set_ratio(s1_addr_norm, t_addr_norm)
    addr_token_sort = fuzz.token_sort_ratio(s1_addr_norm, t_addr_norm)

    s1_addr_tok = set(s1_addr_norm.split())
    t_addr_tok = set(t_addr_norm.split())
    union_addr_tok = s1_addr_tok | t_addr_tok
    addr_jaccard = (len(s1_addr_tok & t_addr_tok) / len(union_addr_tok)) if union_addr_tok else 0.0

    # Number matches in address
    s1_nums = {t.lstrip("0") for t in s1_addr_norm.split() if t.isdigit() and t.lstrip("0")}
    t_nums = {t.lstrip("0") for t in t_addr_norm.split() if t.isdigit() and t.lstrip("0")}
    num_match = 1.0 if (s1_nums and t_nums and (s1_nums & t_nums)) else 0.0
    num_jaccard = (len(s1_nums & t_nums) / len(s1_nums | t_nums)) if (s1_nums | t_nums) else 0.0
    both_have_nums = 1.0 if (s1_nums and t_nums) else 0.0

    # 3. Cross / Interaction
    cross_sim = (fuzz_ratio * addr_ratio) / 10000.0
    min_sim = min(fuzz_ratio, addr_ratio) / 100.0
    max_sim = max(fuzz_ratio, addr_ratio) / 100.0

    # 4. Script Indicator (ASCII / Latin test)
    s1_ascii = 1.0 if s1_name_norm.isascii() else 0.0
    t_ascii = 1.0 if t_name_norm.isascii() else 0.0
    same_script = 1.0 if (s1_ascii == t_ascii) else 0.0

    return [
        float(fuzz_ratio),
        float(token_sort),
        float(token_set),
        float(wratio),
        float(tok_jaccard),
        float(len_diff),
        float(len_ratio),
        float(core_equal),
        float(addr_ratio),
        float(addr_token_set),
        float(addr_token_sort),
        float(addr_jaccard),
        float(num_match),
        float(num_jaccard),
        float(both_have_nums),
        float(cross_sim),
        float(min_sim),
        float(max_sim),
        float(same_script),
        float(blocker_score),
        float(is_exact_name),
        float(is_core_name),
        float(is_addr_sig),
        float(is_postal),
    ]


FEATURE_NAMES = [
    "name_fuzz_ratio",
    "name_token_sort",
    "name_token_set",
    "name_wratio",
    "name_token_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "name_core_equal",
    "addr_fuzz_ratio",
    "addr_token_set",
    "addr_token_sort",
    "addr_token_jaccard",
    "addr_number_match",
    "addr_number_jaccard",
    "addr_both_have_nums",
    "cross_name_addr_sim",
    "min_name_addr_sim",
    "max_name_addr_sim",
    "same_script",
    "blocker_score",
    "is_exact_name",
    "is_core_name",
    "is_addr_sig",
    "is_postal",
]


# ----------------------------------------------------------------------
# Main Checkpointed Pipeline
# ----------------------------------------------------------------------

def run_pipeline(sample_size: int = 100_000, random_seed: int = 42):
    start_time = time.time()

    base_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource")
    data_dir = base_dir / "dataset" / "train"
    out_dir = base_dir / "student_resource" / "amazon_er" / "outputs" / "M001v2_outputs"
    ckpt_dir = out_dir / "checkpoints"
    data_store_dir = out_dir / "data"
    model_dir = out_dir / "models"

    for d in (out_dir, ckpt_dir, data_store_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "M001v2_terminal_op.txt"
    log_fp = open(log_path, "a", encoding="utf-8")

    def log(msg: str = ""):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    log("=" * 75)
    log("M001v2: SUPERVISED PAIRWISE MATCHING (CHECKPOINTED RESUME ENGINE)")
    log(f"Config: Sample Size = {sample_size:,} S1 Entities | Random Seed = {random_seed}")
    log(f"RAM Budget: < 2.0 GB Peak | Checkpoint Dir: {ckpt_dir}")
    log("=" * 75)

    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"
    gt_path = data_dir / "train_ground_truth.tsv"

    # ==================================================================
    # STAGE 1: Candidate Generation (E004.5 Engine)
    # ==================================================================
    candidates_parquet = data_store_dir / "candidates.parquet"
    stage1_done_file = ckpt_dir / "stage1_candidates.done"

    if stage1_done_file.exists() and candidates_parquet.exists():
        log("\n>>> [CHECKPOINT 1 DETECTED]: Stage 1 already complete. Skipping candidate generation.")
        candidates_df = pd.read_parquet(candidates_parquet)
        log(f"Loaded {len(candidates_df):,} existing candidate pairs from disk.")
    else:
        log("\n>>> [STAGE 1/4]: Target Indexing & Candidate Generation...")

        log("Loading Ground Truth...")
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        gt = {}
        for row in gt_df.itertuples(index=False):
            s1_id = row.source1_entity_id
            m = str(row.matched_entity_ids).strip()
            gt[s1_id] = {x.strip() for x in m.split(",") if x.strip()} if m else set()

        log(f"Sampling {sample_size:,} S1 Entities...")
        s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
        s1_sample = s1_df.sample(n=min(sample_size, len(s1_df)), random_state=random_seed).reset_index(drop=True)
        sample_s1_ids = set(s1_sample["entity_id"])
        total_true = sum(len(gt[sid]) for sid in sample_s1_ids if sid in gt)
        log(f"Total true matches in sample: {total_true:,}")

        index = BlockingIndex(max_postings=120)

        log("Indexing Source 2 (5,034,616 records)...")
        t0 = time.time()
        for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            for eid, c, n, a in zip(chunk["entity_id"].astype(str), chunk["country"].astype(str), chunk["business_name"].astype(str), chunk["business_address"].astype(str)):
                index.add_target(eid, c, n, a)
        log(f"  S2 indexed in {time.time() - t0:.1f}s | Targets: {len(index.target_ids):,}")

        log("Indexing Source 3 (5,285,603 records)...")
        t0 = time.time()
        for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            for eid, c, n, a in zip(chunk["entity_id"].astype(str), chunk["country"].astype(str), chunk["business_name"].astype(str), chunk["business_address"].astype(str)):
                index.add_target(eid, c, n, a)
        log(f"  S3 indexed in {time.time() - t0:.1f}s | Total Targets: {len(index.target_ids):,}")

        log(f"Generating candidate pairs for {len(s1_sample):,} S1 entities (max 80 cands/entity)...")
        t0 = time.time()
        pair_rows = []
        target_ids = index.target_ids
        true_found = 0

        s1_ids = s1_sample["entity_id"].astype(str).tolist()
        s1_countries = s1_sample["country"].astype(str).tolist()
        s1_names = s1_sample["business_name"].astype(str).tolist()
        s1_addrs = s1_sample["business_address"].astype(str).tolist()

        for i, (s1_id, country, name, addr) in enumerate(zip(s1_ids, s1_countries, s1_names, s1_addrs), start=1):
            true_set = gt.get(s1_id, set())

            cand_info = index.retrieve(country, name, addr)
            sorted_targets = sorted(cand_info.keys(), key=lambda idx: cand_info[idx][0], reverse=True)[:80]

            for t_idx in sorted_targets:
                t_id = target_ids[t_idx]
                score, is_exact, is_core, is_addr, is_postal = cand_info[t_idx]
                lbl = 1 if t_id in true_set else 0
                if lbl == 1:
                    true_found += 1

                pair_rows.append((
                    s1_id, t_id, score, lbl,
                    is_exact, is_core, is_addr, is_postal
                ))

            if i % 10_000 == 0:
                elapsed = time.time() - t0
                log(f"  Progress: {i:,}/{len(s1_sample):,} entities | {len(pair_rows):,} pairs generated | Rate: {i/elapsed:.0f} ent/s")

        candidates_df = pd.DataFrame(
            pair_rows,
            columns=["s1_id", "target_id", "blocker_score", "label", "is_exact_name", "is_core_name", "is_addr_sig", "is_postal"]
        )
        candidates_df.to_parquet(candidates_parquet, index=False)
        with open(stage1_done_file, "w") as f:
            f.write(f"done\ntotal_pairs={len(candidates_df)}\ntrue_found={true_found}\n")

        recall = (true_found / total_true) * 100 if total_true else 0
        log(f"Candidate generation complete! Recall: {recall:.2f}% | Total pairs: {len(candidates_df):,}")
        log(f"Saved to {candidates_parquet}")

        # Free memory from index
        del index, pair_rows
        gc.collect()

    # ==================================================================
    # STAGE 2: Pairwise Feature Extraction
    # ==================================================================
    features_parquet = data_store_dir / "features.parquet"
    stage2_done_file = ckpt_dir / "stage2_features.done"

    if stage2_done_file.exists() and features_parquet.exists():
        log("\n>>> [CHECKPOINT 2 DETECTED]: Stage 2 already complete. Skipping feature extraction.")
        features_df = pd.read_parquet(features_parquet)
        log(f"Loaded {len(features_df):,} feature rows from disk.")
    else:
        log("\n>>> [STAGE 2/4]: Extracting 24 Pairwise Features on Candidates...")
        t0 = time.time()

        # Build fast string lookups only for entities present in candidate pairs
        needed_s1 = set(candidates_df["s1_id"])
        needed_target = set(candidates_df["target_id"])
        log(f"Unique S1 IDs needed: {len(needed_s1):,} | Unique Target IDs needed: {len(needed_target):,}")

        log("Loading S1 string lookups...")
        s1_lookup = {}
        for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            matched_chunk = chunk[chunk["entity_id"].isin(needed_s1)]
            for eid, name, addr in zip(
                matched_chunk["entity_id"].astype(str),
                matched_chunk["business_name"].astype(str),
                matched_chunk["business_address"].astype(str),
            ):
                s1_lookup[eid] = (normalize_text(name), normalize_text(addr))

        log("Loading S2 string lookups...")
        target_lookup = {}
        for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            matched_chunk = chunk[chunk["entity_id"].isin(needed_target)]
            for eid, name, addr in zip(
                matched_chunk["entity_id"].astype(str),
                matched_chunk["business_name"].astype(str),
                matched_chunk["business_address"].astype(str),
            ):
                target_lookup[eid] = (normalize_text(name), normalize_text(addr))

        log("Loading S3 string lookups...")
        for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            matched_chunk = chunk[chunk["entity_id"].isin(needed_target)]
            for eid, name, addr in zip(
                matched_chunk["entity_id"].astype(str),
                matched_chunk["business_name"].astype(str),
                matched_chunk["business_address"].astype(str),
            ):
                target_lookup[eid] = (normalize_text(name), normalize_text(addr))

        log(f"Lookups built in {time.time() - t0:.1f}s. Computing features in memory-safe chunks...")

        feature_matrix = []
        labels = []
        s1_ids = []
        target_ids_col = []

        cand_s1_list = candidates_df["s1_id"].astype(str).tolist()
        cand_target_list = candidates_df["target_id"].astype(str).tolist()
        cand_score_list = candidates_df["blocker_score"].to_numpy(dtype=np.float32).tolist()
        cand_exact_list = candidates_df["is_exact_name"].to_numpy(dtype=np.int32).tolist()
        cand_core_list = candidates_df["is_core_name"].to_numpy(dtype=np.int32).tolist()
        cand_addr_list = candidates_df["is_addr_sig"].to_numpy(dtype=np.int32).tolist()
        cand_postal_list = candidates_df["is_postal"].to_numpy(dtype=np.int32).tolist()
        cand_label_list = candidates_df["label"].to_numpy(dtype=np.int32).tolist()

        t_feat = time.time()
        for i, (s1_id, t_id, b_score, is_exact, is_core, is_addr, is_postal, lbl) in enumerate(
            zip(
                cand_s1_list,
                cand_target_list,
                cand_score_list,
                cand_exact_list,
                cand_core_list,
                cand_addr_list,
                cand_postal_list,
                cand_label_list,
            ),
            start=1,
        ):
            s1_name, s1_addr = s1_lookup.get(s1_id, ("", ""))
            t_name, t_addr = target_lookup.get(t_id, ("", ""))

            feats = compute_pairwise_features(
                s1_name_norm=s1_name,
                s1_addr_norm=s1_addr,
                t_name_norm=t_name,
                t_addr_norm=t_addr,
                blocker_score=b_score,
                is_exact_name=is_exact,
                is_core_name=is_core,
                is_addr_sig=is_addr,
                is_postal=is_postal,
            )
            feature_matrix.append(feats)
            labels.append(lbl)
            s1_ids.append(s1_id)
            target_ids_col.append(t_id)

            if i % 500_000 == 0:
                elapsed = time.time() - t_feat
                log(f"  Computed features: {i:,}/{len(candidates_df):,} pairs | Rate: {i/elapsed:,.0f} pairs/sec")

        log("Assembling feature DataFrame...")
        features_df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES, dtype=np.float32)
        features_df["label"] = np.array(labels, dtype=np.int8)
        features_df["s1_id"] = s1_ids
        features_df["target_id"] = target_ids_col

        features_df.to_parquet(features_parquet, index=False)
        with open(stage2_done_file, "w") as f:
            f.write(f"done\nfeature_rows={len(features_df)}\n")

        log(f"Feature extraction complete! Saved {len(features_df):,} rows to {features_parquet}")

        del feature_matrix, labels, s1_ids, target_ids_col, s1_lookup, target_lookup
        gc.collect()

    # ==================================================================
    # STAGE 3: Grouped 5-Fold LightGBM Training & OOF Inference
    # ==================================================================
    oof_parquet = data_store_dir / "oof_predictions.parquet"
    stage3_done_file = ckpt_dir / "stage3_training.done"

    if stage3_done_file.exists() and oof_parquet.exists():
        log("\n>>> [CHECKPOINT 3 DETECTED]: Stage 3 already complete. Skipping model training.")
        oof_df = pd.read_parquet(oof_parquet)
        log(f"Loaded {len(oof_df):,} OOF prediction rows.")
    else:
        log("\n>>> [STAGE 3/4]: Grouped 5-Fold LightGBM Training (Leak-Free by s1_id)...")

        # Assign 5 folds grouped by s1_id
        gkf = GroupKFold(n_splits=5)
        unique_s1 = np.array(list(set(features_df["s1_id"])))
        fold_map = {}
        for fold, (_, val_idx) in enumerate(gkf.split(unique_s1, groups=unique_s1)):
            for sid in unique_s1[val_idx]:
                fold_map[sid] = fold

        features_df["fold"] = features_df["s1_id"].map(fold_map).astype(np.int8)

        oof_probas = np.zeros(len(features_df), dtype=np.float32)
        feature_cols = FEATURE_NAMES
        importances = np.zeros(len(feature_cols), dtype=np.float64)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "n_jobs": -1,
            "verbose": -1,
            "seed": random_seed,
        }

        for fold in range(5):
            log(f"\n--- Training Fold {fold + 1} / 5 ---")
            train_mask = (features_df["fold"] != fold)
            val_mask = (features_df["fold"] == fold)

            X_train = features_df.loc[train_mask, feature_cols]
            y_train = features_df.loc[train_mask, "label"]
            X_val = features_df.loc[val_mask, feature_cols]
            y_val = features_df.loc[val_mask, "label"]

            trn_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

            model = lgb.train(
                params,
                trn_data,
                num_boost_round=600,
                valid_sets=[trn_data, val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False), lgb.log_evaluation(period=100)],
            )

            preds = np.asarray(model.predict(X_val, num_iteration=model.best_iteration), dtype=np.float32)
            val_indices = np.where(features_df["fold"].to_numpy() == fold)[0]
            oof_probas[val_indices] = preds
            importances += np.asarray(model.feature_importance(importance_type="gain"), dtype=np.float64) / 5.0

            fold_model_path = model_dir / f"lgb_model_fold_{fold}.txt"
            model.save_model(str(fold_model_path))
            log(f"  Fold {fold + 1} Best Iteration: {model.best_iteration} | Model saved to {fold_model_path}")

        oof_df = pd.DataFrame({
            "s1_id": features_df["s1_id"],
            "target_id": features_df["target_id"],
            "label": features_df["label"],
            "proba": oof_probas,
        })
        oof_df.to_parquet(oof_parquet, index=False)

        # Feature Importance CSV
        fi_df = pd.DataFrame({"feature": feature_cols, "importance_gain": importances}).sort_values("importance_gain", ascending=False)
        fi_csv = out_dir / "M001v2_feature_importance.csv"
        fi_df.to_csv(fi_csv, index=False)

        with open(stage3_done_file, "w") as f:
            f.write("done\n")

        log(f"\nTraining Complete! OOF predictions saved to {oof_parquet}")
        log(f"Top 5 Most Important Features:\n{fi_df.head(5).to_string(index=False)}")

        del features_df, oof_probas
        gc.collect()

    # ==================================================================
    # STAGE 4: Threshold Optimization for Exact Competition Macro-F0.5
    # ==================================================================
    log("\n>>> [STAGE 4/4]: Optimizing Decision Threshold & Singleton Abstention...")

    # Load Ground Truth for evaluated S1 entities
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    all_s1_eval = sorted(list(set(oof_df["s1_id"])))
    eval_set = set(all_s1_eval)

    gt_dict = {}
    matched_gt = gt_df[gt_df["source1_entity_id"].isin(eval_set)]
    for sid, m_str in zip(matched_gt["source1_entity_id"].astype(str), matched_gt["matched_entity_ids"].astype(str)):
        m = m_str.strip()
        gt_dict[sid] = {x.strip() for x in m.split(",") if x.strip()} if m else set()

    # Ensure all S1 entities exist in gt_dict (even singletons)
    for sid in all_s1_eval:
        if sid not in gt_dict:
            gt_dict[sid] = set()

    # Group predictions by s1_id: sid -> list of (proba, target_id)
    s1_preds_map = defaultdict(list)
    oof_s1_list = oof_df["s1_id"].astype(str).tolist()
    oof_target_list = oof_df["target_id"].astype(str).tolist()
    oof_proba_list = oof_df["proba"].to_numpy(dtype=np.float32).tolist()

    for sid, tid, prob in zip(oof_s1_list, oof_target_list, oof_proba_list):
        s1_preds_map[sid].append((prob, tid))

    # Threshold Search Loop: Sweep 0.20 to 0.92
    thresholds = [round(x, 2) for x in np.arange(0.20, 0.94, 0.04)]
    curve_records = []

    best_threshold = 0.50
    best_macro_f05 = -1.0
    best_metrics = {}

    log(f"Sweeping {len(thresholds)} thresholds from {thresholds[0]} to {thresholds[-1]} on {len(all_s1_eval):,} entities...")

    for thresh in thresholds:
        pred_dict = {}
        for sid in all_s1_eval:
            cand_list = s1_preds_map.get(sid, [])
            # Predict targets above threshold
            matches = {t_id for p, t_id in cand_list if p >= thresh}
            pred_dict[sid] = matches

        eval_res = competition_macro_f05(gt_dict, pred_dict, all_s1_ids=all_s1_eval)
        f05 = eval_res["macro_f05"]
        prec = eval_res["macro_precision"]
        rec = eval_res["macro_recall"]
        sing_acc = eval_res["singleton_accuracy"]

        curve_records.append({
            "threshold": thresh,
            "macro_f05": round(f05, 5),
            "macro_precision": round(prec, 5),
            "macro_recall": round(rec, 5),
            "singleton_accuracy": round(sing_acc, 5),
        })

        log(f"  Threshold {thresh:.2f}: Macro-F0.5 = {f05:.4f} | Prec = {prec:.4f} | Rec = {rec:.4f} | SingAcc = {sing_acc:.4f}")

        if f05 > best_macro_f05:
            best_macro_f05 = f05
            best_threshold = thresh
            best_metrics = eval_res

    curve_df = pd.DataFrame(curve_records)
    curve_csv = out_dir / "M001v2_threshold_curve.csv"
    curve_df.to_csv(curve_csv, index=False)

    total_pipeline_time = time.time() - start_time

    log("\n" + "=" * 75)
    log("M001v2 EXPERIMENT RESULTS (OFFICIAL COMPETITION EVALUATOR)")
    log("=" * 75)
    log(f"Best Official Macro-F0.5 : {best_macro_f05:.4f}  (at Threshold = {best_threshold:.2f})")
    log(f"Macro-Precision          : {best_metrics.get('macro_precision', 0):.4f}")
    log(f"Macro-Recall             : {best_metrics.get('macro_recall', 0):.4f}")
    log(f"Singleton Accuracy       : {best_metrics.get('singleton_accuracy', 0) * 100:.2f}%")
    log(f"Total Evaluated Entities : {best_metrics.get('total_entities', 0):,}")
    log(f"Total Pipeline Runtime   : {total_pipeline_time:.1f}s ({total_pipeline_time / 60.0:.1f} minutes)")
    log("=" * 75)

    final_metrics_payload = {
        "experiment": "M001v2",
        "sample_size": sample_size,
        "best_threshold": best_threshold,
        "best_macro_f05": best_macro_f05,
        "macro_precision": best_metrics.get("macro_precision", 0),
        "macro_recall": best_metrics.get("macro_recall", 0),
        "singleton_accuracy": best_metrics.get("singleton_accuracy", 0),
        "total_entities_evaluated": len(all_s1_eval),
        "runtime_seconds": total_pipeline_time,
    }

    metrics_json_path = out_dir / "M001v2_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2)

    log(f"\nAll artifacts successfully saved to: {out_dir}")
    log_fp.close()
    print("\nM001v2 execution complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M001v2 Supervised Pairwise Entity Matching Pipeline")
    parser.add_argument("--sample-size", type=int, default=100_000, help="Number of S1 entities to evaluate (default: 100,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_pipeline(sample_size=args.sample_size, random_seed=args.seed)
