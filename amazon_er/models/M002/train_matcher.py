#!/usr/bin/env python3
"""
M002: Enhanced Supervised Pairwise Entity Resolution Pipeline

Improvements over M001v2:
1. Stage 1: Candidate cap increased to 100 (lifts candidate recall ceiling to 82.5%+).
2. Stage 2: 30-feature pairwise matrix including:
   - Core Name Token-Sort and Token-Set ratios (legal suffixes & domains stripped)
   - Exact House / Building Number Match (numeric anchor)
   - House Number Mismatch Penalty (different numbers on same street)
   - Address Non-generic Word Jaccard
   - Harmonic Core Name & Address interaction feature
3. Stage 3: Tuned 5-Fold Grouped LightGBM (leak-free by S1 entity)
4. Stage 4: Optimal threshold sweep + 1-to-many Bipartite Disambiguation (S2/S3 entity exclusivity)

Run command:
  C:\\Users\\Shardul\\AppData\\Local\\Python\\bin\\python.exe amazon_er/models/M002/train_matcher.py --sample-size 100000
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import GroupKFold

# Ensure package import
current_dir = Path(__file__).resolve().parent
pkg_root = current_dir.parent.parent
sys.path.insert(0, str(pkg_root))
from src.metric import competition_macro_f05


# ----------------------------------------------------------------------
# Constants & Precompilations
# ----------------------------------------------------------------------

LEGAL_SUFFIXES = [
    "pvt ltd", "private limited", "ltd", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "llp", "gmbh", "sa", "sarl", "co", "company",
    "enterprises", "services", "industries", "holdings", "group", "associates",
    "solutions", "technologies", "tech", "international", "mfg", "manufacturing",
    "foundation", "trust", "society", "association", "agency", "firm", "ventures"
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
    "post", "po", "via", "circle", "layout", "extension", "ext", "stage", "rue", "avenue"
}

FEATURE_NAMES = [
    # Lexical Name Similarities
    "name_fuzz_ratio",
    "name_token_sort",
    "name_token_set",
    "name_wratio",
    "name_core_token_sort",
    "name_core_token_set",
    "name_core_equal",
    "name_token_jaccard",
    "name_len_diff",
    "name_len_ratio",
    # Address Similarities
    "addr_fuzz_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_token_jaccard",
    "addr_word_jaccard",
    "addr_number_jaccard",
    "addr_number_match",
    "exact_house_number_match",
    "house_number_mismatch",
    "addr_both_have_nums",
    # Interaction Features
    "min_name_addr_sim",
    "max_name_addr_sim",
    "cross_name_addr_sim",
    "harmonic_core_sim",
    # Blocker Signal Flags
    "blocker_score",
    "is_exact_name",
    "is_core_name",
    "is_addr_sig",
    "is_postal",
    "same_script",
]

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
        for w in valid_words[:4]:
            signatures.append((c, num, w))
    return signatures


def extract_postal_anchors(country: str, address_norm: str) -> list[tuple[str, str, str]]:
    if not address_norm:
        return []
    c = country.casefold()
    tokens = address_norm.split()
    anchors = []

    if c == "in":
        codes = [t for t in tokens if t.isdigit() and len(t) == 6 and t[0] in "123456789"]
    elif c == "us":
        codes = [t for t in tokens if t.isdigit() and len(t) == 5]
    elif c == "fr":
        codes = [t for t in tokens if t.isdigit() and len(t) == 5]
    else:
        codes = [t for t in tokens if t.isdigit() and len(t) in (5, 6)]

    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS]
    for pc in codes[:2]:
        for w in words[:3]:
            anchors.append((c, pc, w))
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


def extract_distinctive_tokens(name_norm: str) -> list[str]:
    tokens = [t for t in name_norm.split() if len(t) >= 6 and not t.isdigit()]
    return tokens[:3]


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


# ----------------------------------------------------------------------
# Compact Blocking Index (Inverted uint32 Postings)
# ----------------------------------------------------------------------

class BlockingIndex:
    def __init__(self, max_postings: int = 150):
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

        if len(cand_info) < 30:
            for p1, p2 in extract_prefix_pairs(name_norm):
                k = (c, p1, p2)
                if k in self.idx_prefix_pairs:
                    lst = self.idx_prefix_pairs[k]
                    if len(lst) <= 60:
                        w = SIGNAL_WEIGHTS["prefix_pairs"]
                        for idx in lst:
                            cand_info[idx][0] += w

        if len(cand_info) < 20:
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
# M002 Enhanced 30 Pairwise Feature Extraction
# ----------------------------------------------------------------------

def compute_pairwise_features_m002(
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
    """Computes rich 30-feature vector for S1-Target pair."""
    # 1. Lexical Name Similarities
    fuzz_ratio = fuzz.ratio(s1_name_norm, t_name_norm) / 100.0
    tok_sort = fuzz.token_sort_ratio(s1_name_norm, t_name_norm) / 100.0
    tok_set = fuzz.token_set_ratio(s1_name_norm, t_name_norm) / 100.0
    wratio = fuzz.WRatio(s1_name_norm, t_name_norm) / 100.0

    # Core name similarities (legal suffixes stripped)
    s1_core = extract_core_name(s1_name_norm)
    t_core = extract_core_name(t_name_norm)
    core_equal = 1.0 if s1_core and t_core and (s1_core == t_core) else 0.0
    core_sort = fuzz.token_sort_ratio(s1_core, t_core) / 100.0 if (s1_core and t_core) else tok_sort
    core_set = fuzz.token_set_ratio(s1_core, t_core) / 100.0 if (s1_core and t_core) else tok_set

    # Name token jaccard
    s1_ntoks = set(s1_name_norm.split())
    t_ntoks = set(t_name_norm.split())
    name_jaccard = len(s1_ntoks & t_ntoks) / len(s1_ntoks | t_ntoks) if (s1_ntoks | t_ntoks) else 0.0

    len1, len2 = len(s1_name_norm), len(t_name_norm)
    name_len_diff = abs(len1 - len2)
    name_len_ratio = min(len1, len2) / max(len1, len2, 1)

    # 2. Address Similarities
    addr_fuzz = fuzz.ratio(s1_addr_norm, t_addr_norm) / 100.0
    addr_sort = fuzz.token_sort_ratio(s1_addr_norm, t_addr_norm) / 100.0
    addr_set = fuzz.token_set_ratio(s1_addr_norm, t_addr_norm) / 100.0

    s1_atoks = set(s1_addr_norm.split())
    t_atoks = set(t_addr_norm.split())
    addr_jaccard = len(s1_atoks & t_atoks) / len(s1_atoks | t_atoks) if (s1_atoks | t_atoks) else 0.0

    # Address Non-generic word jaccard
    s1_words = {w for w in s1_atoks if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_GENERIC_WORDS}
    t_words = {w for w in t_atoks if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_GENERIC_WORDS}
    addr_word_jaccard = len(s1_words & t_words) / len(s1_words | t_words) if (s1_words | t_words) else 0.0

    # Numbers in addresses
    s1_nums = [t.lstrip("0") for t in s1_atoks if t.isdigit() and len(t) <= 7]
    t_nums = [t.lstrip("0") for t in t_atoks if t.isdigit() and len(t) <= 7]
    s1_nums_set = set(s1_nums)
    t_nums_set = set(t_nums)

    both_nums = 1.0 if (s1_nums_set and t_nums_set) else 0.0
    num_jaccard = len(s1_nums_set & t_nums_set) / len(s1_nums_set | t_nums_set) if (s1_nums_set | t_nums_set) else 0.0
    num_match = 1.0 if (s1_nums_set & t_nums_set) else 0.0

    # First numeric token (house/building number)
    s1_first_num = s1_nums[0] if s1_nums else ""
    t_first_num = t_nums[0] if t_nums else ""
    exact_house_match = 1.0 if (s1_first_num and t_first_num and s1_first_num == t_first_num) else 0.0
    house_mismatch = 1.0 if (s1_first_num and t_first_num and s1_first_num != t_first_num) else 0.0

    # 3. Cross & Interaction Features
    min_sim = min(wratio, addr_sort)
    max_sim = max(wratio, addr_sort)
    cross_sim = (2.0 * wratio * addr_sort) / (wratio + addr_sort) if (wratio + addr_sort) > 0 else 0.0
    harmonic_core = (2.0 * core_sort * addr_sort) / (core_sort + addr_sort) if (core_sort + addr_sort) > 0 else 0.0

    # 4. Script Agreement
    s1_scr = detect_script(s1_name_norm)
    t_scr = detect_script(t_name_norm)
    same_script = 1.0 if s1_scr == t_scr else 0.0

    return [
        fuzz_ratio,
        tok_sort,
        tok_set,
        wratio,
        core_sort,
        core_set,
        core_equal,
        name_jaccard,
        float(name_len_diff),
        name_len_ratio,
        addr_fuzz,
        addr_sort,
        addr_set,
        addr_jaccard,
        addr_word_jaccard,
        num_jaccard,
        num_match,
        exact_house_match,
        house_mismatch,
        both_nums,
        min_sim,
        max_sim,
        cross_sim,
        harmonic_core,
        blocker_score,
        float(is_exact_name),
        float(is_core_name),
        float(is_addr_sig),
        float(is_postal),
        same_script,
    ]


# ----------------------------------------------------------------------
# Main Checkpointed Pipeline
# ----------------------------------------------------------------------

def run_pipeline(sample_size: int = 100_000, random_seed: int = 42):
    start_time = time.time()

    base_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource")
    data_dir = base_dir / "dataset" / "train"
    out_dir = base_dir / "student_resource" / "amazon_er" / "outputs" / "M002_outputs"
    ckpt_dir = out_dir / "checkpoints"
    data_store_dir = out_dir / "data"
    model_dir = out_dir / "models"

    for d in (out_dir, ckpt_dir, data_store_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "M002_terminal_op.txt"
    log_fp = open(log_path, "a", encoding="utf-8")

    def log(msg: str = ""):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    log("=" * 75)
    log("M002: ENHANCED PAIRWISE MATCHING & BIPARTITE DISAMBIGUATION")
    log(f"Config: Sample Size = {sample_size:,} S1 Entities | Random Seed = {random_seed}")
    log(f"RAM Budget: < 2.0 GB Peak | Checkpoint Dir: {ckpt_dir}")
    log("=" * 75)

    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"
    gt_path = data_dir / "train_ground_truth.tsv"

    candidates_parquet = data_store_dir / "candidates.parquet"
    features_parquet = data_store_dir / "features.parquet"
    oof_parquet = data_store_dir / "oof_predictions.parquet"

    stage1_done_file = ckpt_dir / "stage1_candidates.done"
    stage2_done_file = ckpt_dir / "stage2_features.done"
    stage3_done_file = ckpt_dir / "stage3_training.done"

    # ==================================================================
    # STAGE 1: Target Indexing & Candidate Generation (Cap = 100)
    # ==================================================================
    if stage1_done_file.exists() and candidates_parquet.exists():
        log(">>> [STAGE 1/4]: Found existing candidates checkpoint. Loading...")
        candidates_df = pd.read_parquet(candidates_parquet)
        log(f"  Loaded {len(candidates_df):,} candidates from {candidates_parquet}")
    else:
        log("\n>>> [STAGE 1/4]: Target Indexing & Candidate Generation (Cap = 100)...")
        log("Loading Ground Truth...")
        gt = defaultdict(set)
        for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            for sid, m in zip(chunk["source1_entity_id"].astype(str), chunk["matched_entity_ids"].astype(str)):
                m_str = m.strip()
                if m_str:
                    gt[sid] = {x.strip() for x in m_str.split(",") if x.strip()}

        log(f"Sampling {sample_size:,} S1 Entities...")
        s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
        s1_sample = s1_df.sample(n=min(sample_size, len(s1_df)), random_state=random_seed).reset_index(drop=True)
        sample_s1_ids = set(s1_sample["entity_id"])
        total_true = sum(len(gt[sid]) for sid in sample_s1_ids if sid in gt)
        log(f"Total true matches in sample: {total_true:,}")

        index = BlockingIndex(max_postings=150)

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

        log(f"Generating candidate pairs for {len(s1_sample):,} S1 entities (max 100 cands/entity)...")
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
            sorted_targets = sorted(cand_info.keys(), key=lambda idx: cand_info[idx][0], reverse=True)[:100]

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

        del index, pair_rows
        gc.collect()

    # ==================================================================
    # STAGE 2: Extracting 30 Enhanced Pairwise Features
    # ==================================================================
    if stage2_done_file.exists() and features_parquet.exists():
        log("\n>>> [STAGE 2/4]: Found existing features checkpoint. Loading...")
        features_df = pd.read_parquet(features_parquet)
        log(f"  Loaded {len(features_df):,} feature rows from {features_parquet}")
    else:
        log("\n>>> [STAGE 2/4]: Extracting 30 Enhanced Pairwise Features on Candidates...")
        t0 = time.time()

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

            feats = compute_pairwise_features_m002(
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

        log(f"Saved {len(features_df):,} feature rows to {features_parquet}")
        del s1_lookup, target_lookup, feature_matrix, labels, s1_ids, target_ids_col
        del cand_s1_list, cand_target_list, cand_score_list, cand_exact_list, cand_core_list, cand_addr_list, cand_postal_list, cand_label_list
        gc.collect()

    # ==================================================================
    # STAGE 3: Grouped 5-Fold LightGBM Training (Leak-Free by S1 entity)
    # ==================================================================
    if stage3_done_file.exists() and oof_parquet.exists():
        log("\n>>> [STAGE 3/4]: Found existing training checkpoint. Loading OOF...")
        oof_df = pd.read_parquet(oof_parquet)
    else:
        log("\n>>> [STAGE 3/4]: Grouped 5-Fold LightGBM Training (Leak-Free by S1 entity)...")

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
            "learning_rate": 0.04,
            "num_leaves": 45,
            "max_depth": 7,
            "min_child_samples": 30,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
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
                num_boost_round=800,
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
        fi_csv = out_dir / "M002_feature_importance.csv"
        fi_df.to_csv(fi_csv, index=False)

        with open(stage3_done_file, "w") as f:
            f.write("done\n")

        log(f"\nTraining Complete! OOF predictions saved to {oof_parquet}")
        log(f"Top 7 Most Important Features:\n{fi_df.head(7).to_string(index=False)}")

        del features_df, oof_probas
        gc.collect()

    # ==================================================================
    # STAGE 4: Threshold Optimization & Bipartite Disambiguation
    # ==================================================================
    log("\n>>> [STAGE 4/4]: Optimizing Decision Threshold & Bipartite Disambiguation...")

    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    all_s1_eval = sorted(list(set(oof_df["s1_id"])))
    eval_set = set(all_s1_eval)

    gt_dict = {}
    matched_gt = gt_df[gt_df["source1_entity_id"].isin(eval_set)]
    for sid, m_str in zip(matched_gt["source1_entity_id"].astype(str), matched_gt["matched_entity_ids"].astype(str)):
        m = m_str.strip()
        gt_dict[sid] = {x.strip() for x in m.split(",") if x.strip()} if m else set()

    for sid in all_s1_eval:
        if sid not in gt_dict:
            gt_dict[sid] = set()

    oof_s1_list = oof_df["s1_id"].astype(str).tolist()
    oof_target_list = oof_df["target_id"].astype(str).tolist()
    oof_proba_list = oof_df["proba"].to_numpy(dtype=np.float32).tolist()

    # Sort all pairs by probability descending for stable greedy bipartite matching
    log("Running threshold search with and without Bipartite Disambiguation...")
    thresholds = [round(x, 2) for x in np.arange(0.30, 0.86, 0.04)]
    curve_records = []

    best_threshold = 0.50
    best_macro_f05 = -1.0
    best_metrics = {}
    best_mode = "standard"

    for thresh in thresholds:
        # 1. Standard Independent Thresholding
        pred_dict_std = defaultdict(set)
        for sid, tid, prob in zip(oof_s1_list, oof_target_list, oof_proba_list):
            if prob >= thresh:
                pred_dict_std[sid].add(tid)

        res_std = competition_macro_f05(gt_dict, pred_dict_std, all_s1_ids=all_s1_eval)
        f05_std = res_std["macro_f05"]

        # 2. Bipartite Disambiguation (S2/S3 entity assigned to highest-confidence S1 entity)
        # Greedily assign target_id to the single S1 with the highest prob >= thresh
        pred_dict_bip = defaultdict(set)
        target_claimed = {}  # target_id -> (prob, sid)

        for sid, tid, prob in zip(oof_s1_list, oof_target_list, oof_proba_list):
            if prob >= thresh:
                if tid not in target_claimed or prob > target_claimed[tid][0]:
                    target_claimed[tid] = (prob, sid)

        for tid, (prob, sid) in target_claimed.items():
            pred_dict_bip[sid].add(tid)

        res_bip = competition_macro_f05(gt_dict, pred_dict_bip, all_s1_ids=all_s1_eval)
        f05_bip = res_bip["macro_f05"]

        curve_records.append({
            "threshold": thresh,
            "standard_f05": round(f05_std, 5),
            "bipartite_f05": round(f05_bip, 5),
            "standard_prec": round(res_std["macro_precision"], 5),
            "bipartite_prec": round(res_bip["macro_precision"], 5),
            "standard_rec": round(res_std["macro_recall"], 5),
            "bipartite_rec": round(res_bip["macro_recall"], 5),
        })

        log(f"  Thresh {thresh:.2f}: Standard F0.5 = {f05_std:.4f} | Bipartite F0.5 = {f05_bip:.4f} (Prec: {res_bip['macro_precision']:.4f})")

        if f05_bip > best_macro_f05:
            best_macro_f05 = f05_bip
            best_threshold = thresh
            best_metrics = res_bip
            best_mode = "bipartite"

        if f05_std > best_macro_f05:
            best_macro_f05 = f05_std
            best_threshold = thresh
            best_metrics = res_std
            best_mode = "standard"

    curve_df = pd.DataFrame(curve_records)
    curve_csv = out_dir / "M002_threshold_curve.csv"
    curve_df.to_csv(curve_csv, index=False)

    total_pipeline_time = time.time() - start_time

    log("\n" + "=" * 75)
    log("M002 EXPERIMENT RESULTS (OFFICIAL COMPETITION EVALUATOR)")
    log("=" * 75)
    log(f"Best Official Macro-F0.5 : {best_macro_f05:.4f}  (Mode: {best_mode}, Threshold = {best_threshold:.2f})")
    log(f"Macro-Precision          : {best_metrics.get('macro_precision', 0):.4f}")
    log(f"Macro-Recall             : {best_metrics.get('macro_recall', 0):.4f}")
    log(f"Singleton Accuracy       : {best_metrics.get('singleton_accuracy', 0) * 100:.2f}%")
    log(f"Total Evaluated Entities : {best_metrics.get('total_entities', 0):,}")
    log(f"Total Pipeline Runtime   : {total_pipeline_time:.1f}s ({total_pipeline_time / 60.0:.1f} minutes)")
    log("=" * 75)

    final_metrics_payload = {
        "experiment": "M002",
        "sample_size": sample_size,
        "best_mode": best_mode,
        "best_threshold": best_threshold,
        "best_macro_f05": best_macro_f05,
        "macro_precision": best_metrics.get("macro_precision", 0),
        "macro_recall": best_metrics.get("macro_recall", 0),
        "singleton_accuracy": best_metrics.get("singleton_accuracy", 0),
        "total_entities_evaluated": len(all_s1_eval),
        "runtime_seconds": total_pipeline_time,
    }

    metrics_json_path = out_dir / "M002_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2)

    log(f"\nAll artifacts successfully saved to: {out_dir}")
    log_fp.close()
    print("\nM002 execution complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M002 Enhanced Supervised Pairwise Entity Matching Pipeline")
    parser.add_argument("--sample-size", type=int, default=100_000, help="Number of S1 entities to evaluate (default: 100,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_pipeline(sample_size=args.sample_size, random_seed=args.seed)
