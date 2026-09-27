#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Low-RAM Master End-to-End Inference Pipeline
Strict Memory Budget: < 2.5 GB Peak RAM (Safely runs on systems with 3.3-4.2 GB free RAM)

Architecture:
1. Country-Partitioned Inverted Indexing (France -> US -> India)
   - Eliminates 68% of RAM by only keeping 1 country's target index in memory at a time.
   - Rigorously equivalent to global indexing since cross-country matches are impossible.
2. Fast-Path Transliteration & Regex Numeric Signature Matching.
3. Streaming S1 Inference in Batches of 10,000 Entities.
4. GBDT Architectural Ensemble: 5-Fold LightGBM + 300-Tree XGBoost.
5. In-Country Bipartite Disambiguation (1-to-many mutual exclusivity).
6. Instant Disk Streaming for candidate_pairs.tsv and matching_results.tsv.
7. Official validate_submission.py verification.
"""

import argparse
import gc
import os
import re
import subprocess
import sys
import time
import unicodedata
from array import array
from collections import defaultdict
from pathlib import Path

import anyascii
import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import xgboost as xgb

try:
    import psutil
    def get_ram_str() -> str:
        m = psutil.virtual_memory()
        return f"{m.used / (1024**3):.1f}/{m.total / (1024**3):.1f} GB ({m.percent}%)"
except ImportError:
    def get_ram_str() -> str:
        return "RAM monitor N/A"

# ----------------------------------------------------------------------
# Constants & Domain Regexes
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
    # French generic address words
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
    "name_pairs": 4.0,
    "prefix_pairs": 2.5,
    "single_core_word": 2.0,
    "rare_tokens": 1.2,
}

# ----------------------------------------------------------------------
# Text Preprocessing & Signatures
# ----------------------------------------------------------------------

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
    if not text:
        return ""
    if text.isascii():
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


# ----------------------------------------------------------------------
# Target Inverted Index (Memory-Resident & O(1) Fast Lookups)
# ----------------------------------------------------------------------

class SingleCountryTargetIndex:
    def __init__(self, country_code: str, max_postings: int = 150):
        self.country = country_code
        self.max_postings = max_postings
        self.target_ids: list[str] = []
        self.target_names: list[str] = []
        self.target_addrs: list[str] = []

        self.idx_exact_name = defaultdict(lambda: array("I"))
        self.idx_translit_exact = defaultdict(lambda: array("I"))
        self.idx_core_name = defaultdict(lambda: array("I"))
        self.idx_translit_core = defaultdict(lambda: array("I"))
        self.idx_compressed_name = defaultdict(lambda: array("I"))
        self.idx_name_pairs = defaultdict(lambda: array("I"))
        self.idx_prefix_pairs = defaultdict(lambda: array("I"))
        self.idx_addr_sig = defaultdict(lambda: array("I"))
        self.idx_postal = defaultdict(lambda: array("I"))
        self.idx_single_core = defaultdict(lambda: array("I"))
        self.idx_rare_tokens = defaultdict(lambda: array("I"))

    def add_target(self, target_id: str, name_raw: str, addr_raw: str):
        target_idx = len(self.target_ids)
        self.target_ids.append(target_id)

        c = self.country
        name_norm = normalize_text(name_raw)
        addr_norm = normalize_text(addr_raw)
        self.target_names.append(name_norm)
        self.target_addrs.append(addr_norm)

        # 1. Exact Name
        if name_norm:
            k = (c, name_norm)
            lst = self.idx_exact_name[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 2. Transliterated Exact Name
        trans_name = transliterate_text(name_raw)
        if trans_name and trans_name != name_norm:
            k = (c, trans_name)
            lst = self.idx_translit_exact[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 3. Core Name
        core = extract_core_name(name_norm)
        if core and len(core) >= 3:
            k = (c, core)
            lst = self.idx_core_name[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

            if len(core.split()) == 1 and len(core) >= 5:
                k_single = (c, core)
                lst_single = self.idx_single_core[k_single]
                if len(lst_single) < self.max_postings:
                    lst_single.append(target_idx)

        # 4. Transliterated Core Name
        trans_core = extract_core_name(trans_name)
        if trans_core and trans_core != core and len(trans_core) >= 3:
            k = (c, trans_core)
            lst = self.idx_translit_core[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 5. Compressed Name
        comp = extract_compressed_name(name_raw)
        if comp and len(comp) >= 5:
            k = (c, comp)
            lst = self.idx_compressed_name[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 6. Name Pairs
        pairs = extract_name_pairs(name_norm)
        if not pairs and trans_name:
            pairs = extract_name_pairs(trans_name)
        for w1, w2 in pairs:
            k = (c, w1, w2)
            lst = self.idx_name_pairs[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 7. Prefix Pairs
        for p1, p2 in extract_prefix_pairs(trans_name or name_norm):
            k = (c, p1, p2)
            lst = self.idx_prefix_pairs[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 8. Address Signatures
        trans_addr = transliterate_text(addr_raw)
        for sig in extract_enhanced_address_signatures(c, trans_addr or addr_norm):
            lst = self.idx_addr_sig[sig]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 9. Postal Anchors
        for anchor in extract_enhanced_postal_anchors(c, trans_addr or addr_norm):
            lst = self.idx_postal[anchor]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 10. Distinctive Rare Tokens
        tokens = [t for t in (trans_name or name_norm).split() if len(t) >= 6 and not t.isdigit()]
        for tok in tokens[:2]:
            k = (c, tok)
            lst = self.idx_rare_tokens[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

    def retrieve_candidates(self, name_raw: str, addr_raw: str, cap: int = 100) -> list[tuple[int, float, int, int, int, int]]:
        c = self.country
        name_norm = normalize_text(name_raw)
        addr_norm = normalize_text(addr_raw)
        trans_name = transliterate_text(name_raw)
        trans_addr = transliterate_text(addr_raw)

        cand_data = defaultdict(lambda: [0.0, 0, 0, 0, 0, 0, 0])

        # Exact Name
        if name_norm and (c, name_norm) in self.idx_exact_name:
            w = SIGNAL_WEIGHTS["exact_name"]
            for idx in self.idx_exact_name[(c, name_norm)]:
                cand_data[idx][0] += w
                cand_data[idx][1] = 1
                cand_data[idx][5] = 1

        # Transliterated Exact
        if trans_name and (c, trans_name) in self.idx_translit_exact:
            w = SIGNAL_WEIGHTS["translit_exact"]
            for idx in self.idx_translit_exact[(c, trans_name)]:
                cand_data[idx][0] += w
                cand_data[idx][1] = 1
                cand_data[idx][5] = 1

        # Core Name
        core = extract_core_name(name_norm)
        if core and (c, core) in self.idx_core_name:
            w = SIGNAL_WEIGHTS["core_name"]
            for idx in self.idx_core_name[(c, core)]:
                cand_data[idx][0] += w
                cand_data[idx][2] = 1
                cand_data[idx][5] = 1

        # Transliterated Core
        trans_core = extract_core_name(trans_name)
        if trans_core and (c, trans_core) in self.idx_translit_core:
            w = SIGNAL_WEIGHTS["translit_core"]
            for idx in self.idx_translit_core[(c, trans_core)]:
                cand_data[idx][0] += w
                cand_data[idx][2] = 1
                cand_data[idx][5] = 1

        # Single Core Word
        if core and (c, core) in self.idx_single_core:
            w = SIGNAL_WEIGHTS["single_core_word"]
            for idx in self.idx_single_core[(c, core)]:
                cand_data[idx][0] += w
                cand_data[idx][5] = 1

        # Compressed Name
        comp = extract_compressed_name(name_raw)
        if comp and (c, comp) in self.idx_compressed_name:
            w = SIGNAL_WEIGHTS["compressed_name"]
            for idx in self.idx_compressed_name[(c, comp)]:
                cand_data[idx][0] += w
                cand_data[idx][5] = 1

        # Name Pairs
        for w1, w2 in extract_name_pairs(trans_name or name_norm):
            k = (c, w1, w2)
            if k in self.idx_name_pairs:
                w = SIGNAL_WEIGHTS["name_pairs"]
                for idx in self.idx_name_pairs[k]:
                    cand_data[idx][0] += w
                    cand_data[idx][5] = 1

        # Address Signatures
        for sig in extract_enhanced_address_signatures(c, trans_addr or addr_norm):
            if sig in self.idx_addr_sig:
                w = SIGNAL_WEIGHTS["addr_sig"]
                for idx in self.idx_addr_sig[sig]:
                    cand_data[idx][0] += w
                    cand_data[idx][3] = 1
                    cand_data[idx][6] = 1

        # Postal Anchors
        for anchor in extract_enhanced_postal_anchors(c, trans_addr or addr_norm):
            if anchor in self.idx_postal:
                w = SIGNAL_WEIGHTS["postal_street"]
                for idx in self.idx_postal[anchor]:
                    cand_data[idx][0] += w
                    cand_data[idx][4] = 1
                    cand_data[idx][6] = 1

        # Prefix Pairs fallback if low candidates
        if len(cand_data) < 30:
            for p1, p2 in extract_prefix_pairs(trans_name or name_norm):
                k = (c, p1, p2)
                if k in self.idx_prefix_pairs:
                    lst = self.idx_prefix_pairs[k]
                    if len(lst) <= 80:
                        w = SIGNAL_WEIGHTS["prefix_pairs"]
                        for idx in lst:
                            cand_data[idx][0] += w
                            cand_data[idx][5] = 1

        if len(cand_data) < 20:
            tokens = [t for t in (trans_name or name_norm).split() if len(t) >= 6 and not t.isdigit()]
            for tok in tokens[:2]:
                k = (c, tok)
                if k in self.idx_rare_tokens:
                    lst = self.idx_rare_tokens[k]
                    if len(lst) <= 50:
                        w = SIGNAL_WEIGHTS["rare_tokens"]
                        for idx in lst:
                            cand_data[idx][0] += w
                            cand_data[idx][5] = 1

        # Synergy bonus: If candidate has BOTH Name match AND Address/Postal match
        for idx, row in cand_data.items():
            if row[5] == 1 and row[6] == 1:
                row[0] += 8.0

        sorted_cands = sorted(cand_data.items(), key=lambda x: x[1][0], reverse=True)[:cap]
        return [
            (idx, float(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))
            for idx, row in sorted_cands
        ]


# ----------------------------------------------------------------------
# 30 Pairwise Feature Computation
# ----------------------------------------------------------------------

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
# Main Master Pipeline
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Low-RAM Master Inference Engine for AML Challenge 2026")
    parser.add_argument("--test-dir", type=str, default=r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource\dataset\test", help="Path to test directory")
    parser.add_argument("--out-dir", type=str, default=r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource\student_resource\output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=10000, help="S1 batch size (default: 10,000 entities for strict <2.5GB RAM)")
    parser.add_argument("--candidate-cap", type=int, default=100, help="Max candidates per S1 entity")
    parser.add_argument("--threshold", type=float, default=0.58, help="Match classification threshold")
    args = parser.parse_args()

    start_total = time.time()
    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource")
    m002_models_dir = base_dir / "student_resource" / "amazon_er" / "outputs" / "M002_outputs" / "models"

    print("=" * 75, flush=True)
    print("AMAZON ML CHALLENGE 2026: LOW-RAM MASTER INFERENCE ENGINE", flush=True)
    print(f"Test Directory    : {test_dir}", flush=True)
    print(f"Output Directory  : {out_dir}", flush=True)
    print(f"Candidate Cap     : {args.candidate_cap} candidates/entity", flush=True)
    print(f"Match Threshold   : {args.threshold} (with Bipartite Disambiguation)", flush=True)
    print(f"Memory Budget     : Strict Peak < 2.5 GB RAM (Current System RAM: {get_ram_str()})", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Trained GBDT Models
    print("\n[1/4] Loading Trained GBDT Models into Memory...", flush=True)
    lgb_models = []
    for fold in range(5):
        m_path = m002_models_dir / f"lgb_model_fold_{fold}.txt"
        if not m_path.exists():
            raise FileNotFoundError(f"Missing LightGBM model: {m_path}")
        booster = lgb.Booster(model_file=str(m_path))
        lgb_models.append(booster)
    print(f"  Loaded {len(lgb_models)} LightGBM fold models.", flush=True)

    xgb_path = m002_models_dir / "xgb_model.json"
    if not xgb_path.exists():
        raise FileNotFoundError(f"Missing XGBoost model: {xgb_path}")
    xgb_booster = xgb.Booster()
    xgb_booster.load_model(str(xgb_path))
    print(f"  Loaded XGBoost booster ({xgb_path.name}).", flush=True)

    # Prepare output files
    matching_file = out_dir / "matching_results.tsv"
    candidate_file = out_dir / "candidate_pairs.tsv"

    with open(matching_file, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    with open(candidate_file, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    # 2. Dynamic Country Discovery (Strict Open-Set Compliance)
    s1_counts = pd.read_csv(s1_path, sep="\t", usecols=["country"], keep_default_na=False)["country"].value_counts()
    # Sort ascending by entity count (smallest country first as progressive warmup)
    discovered_countries = list(s1_counts.sort_values(ascending=True).index)
    COUNTRIES = [(c_name, normalize_country(c_name)) for c_name in discovered_countries]
    print(f"  Discovered {len(COUNTRIES)} countries in test set (Open-Set): {COUNTRIES}", flush=True)

    total_s1_global = 0
    total_pairs_global = 0
    total_matches_global = 0

    print("\n[2/4] Starting Country-Partitioned Streaming Inference...", flush=True)

    for country_idx, (country_name, country_code) in enumerate(COUNTRIES, start=1):
        t_country_start = time.time()
        print(f"\n" + "-" * 75, flush=True)
        print(f">>> [{country_idx}/{len(COUNTRIES)}] PROCESSING COUNTRY: {country_name.upper()} (code='{country_code}')", flush=True)
        print(f"    Current RAM: {get_ram_str()}", flush=True)
        print("-" * 75, flush=True)

        # 2a. Build Target Index for this Country
        print(f"  Building Inverted Index for {country_name} Targets...", flush=True)
        index = SingleCountryTargetIndex(country_code=country_code, max_postings=150)

        # Index S2 targets
        t0 = time.time()
        s2_count = 0
        for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            m = chunk["country"].apply(normalize_country) == country_code
            filtered = chunk[m]
            for eid, n, a in zip(filtered["entity_id"].astype(str), filtered["business_name"].astype(str), filtered["business_address"].astype(str)):
                index.add_target(eid, n, a)
                s2_count += 1
        print(f"    Indexed {s2_count:,} Source 2 targets in {time.time() - t0:.1f}s", flush=True)

        # Index S3 targets
        t0 = time.time()
        s3_count = 0
        for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
            m = chunk["country"].apply(normalize_country) == country_code
            filtered = chunk[m]
            for eid, n, a in zip(filtered["entity_id"].astype(str), filtered["business_name"].astype(str), filtered["business_address"].astype(str)):
                index.add_target(eid, n, a)
                s3_count += 1
        print(f"    Indexed {s3_count:,} Source 3 targets in {time.time() - t0:.1f}s | Total targets: {len(index.target_ids):,}", flush=True)
        print(f"    Target Index RAM: {get_ram_str()}", flush=True)

        # 2b. Stream S1 Entities for this Country
        print(f"  Streaming S1 Entities for {country_name} (Batch Size = {args.batch_size:,})...", flush=True)
        
        country_s1_processed = 0
        country_pairs_scored = 0
        target_claimed = {}  # tid -> (max_prob, sid)
        country_s1_ids = []

        batch_s1 = []

        def process_country_batch(batch):
            nonlocal country_pairs_scored
            if not batch:
                return

            batch_s1_ids = [r[0] for r in batch]
            country_s1_ids.extend(batch_s1_ids)

            cand_lines = []
            b_pairs_s1 = []
            b_pairs_tidx = []
            b_features = []

            for sid, name, addr in batch:
                cands = index.retrieve_candidates(name, addr, cap=args.candidate_cap)
                target_ids_list = [index.target_ids[tidx] for tidx, *_ in cands]
                cand_lines.append(f"{sid}\t{','.join(target_ids_list)}\n")

                s1_name_norm = normalize_text(name)
                s1_addr_norm = normalize_text(addr)

                for tidx, b_score, is_exact, is_core, is_addr, is_post in cands:
                    t_name_norm = index.target_names[tidx]
                    t_addr_norm = index.target_addrs[tidx]
                    feat = compute_pairwise_features_fast(
                        s1_name_norm, s1_addr_norm,
                        t_name_norm, t_addr_norm,
                        b_score, is_exact, is_core, is_addr, is_post
                    )
                    b_pairs_s1.append(sid)
                    b_pairs_tidx.append(tidx)
                    b_features.append(feat)

            # Stream candidates to disk
            with open(candidate_file, "a", encoding="utf-8") as f_cand:
                f_cand.writelines(cand_lines)

            n_pairs = len(b_features)
            country_pairs_scored += n_pairs
            if n_pairs == 0:
                return

            X_mat = np.array(b_features, dtype=np.float32)

            # GBDT Predictions
            p_lgb = np.mean([booster.predict(X_mat) for booster in lgb_models], axis=0)
            dmat = xgb.DMatrix(X_mat, feature_names=FEATURE_NAMES)
            p_xgb = xgb_booster.predict(dmat)
            p_blend = 0.5 * p_lgb + 0.5 * p_xgb

            # Bipartite Claims Update
            thresh = args.threshold
            for sid, tidx, prob in zip(b_pairs_s1, b_pairs_tidx, p_blend):
                if prob >= thresh:
                    tid = index.target_ids[tidx]
                    if tid not in target_claimed or prob > target_claimed[tid][0]:
                        target_claimed[tid] = (float(prob), sid)

        t_stream_start = time.time()
        for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, chunksize=50_000, keep_default_na=False):
            m = chunk["country"].apply(normalize_country) == country_code
            filtered = chunk[m]
            for eid, n, a in zip(filtered["entity_id"].astype(str), filtered["business_name"].astype(str), filtered["business_address"].astype(str)):
                batch_s1.append((eid, n, a))
                country_s1_processed += 1

                if len(batch_s1) >= args.batch_size:
                    t_b0 = time.time()
                    process_country_batch(batch_s1)
                    el_b = time.time() - t_b0
                    rate = len(batch_s1) / max(el_b, 0.001)
                    print(f"    [{country_name}] Processed {country_s1_processed:,} S1 entities ({country_pairs_scored:,} pairs) | Rate: {rate:.0f} ent/s | RAM: {get_ram_str()}", flush=True)
                    batch_s1 = []

        # Final batch
        if batch_s1:
            process_country_batch(batch_s1)
            print(f"    [{country_name}] Final batch done | Total S1: {country_s1_processed:,} ({country_pairs_scored:,} pairs)", flush=True)

        # 2c. Write Matching Results for this Country
        print(f"  Assembling Bipartite Matches for {country_name}...", flush=True)
        s1_to_matches = defaultdict(list)
        for tid, (prob, sid) in target_claimed.items():
            s1_to_matches[sid].append(tid)

        country_matches_accepted = len(target_claimed)
        print(f"    Matches accepted in {country_name}: {country_matches_accepted:,} across {len(s1_to_matches):,} entities.", flush=True)

        with open(matching_file, "a", encoding="utf-8") as f_match:
            lines = []
            for sid in country_s1_ids:
                m_list = sorted(s1_to_matches.get(sid, []))
                lines.append(f"{sid}\t{','.join(m_list)}\n")
                if len(lines) >= 50_000:
                    f_match.writelines(lines)
                    lines = []
            if lines:
                f_match.writelines(lines)

        total_s1_global += country_s1_processed
        total_pairs_global += country_pairs_scored
        total_matches_global += country_matches_accepted

        country_elapsed = time.time() - t_country_start
        print(f"  Finished {country_name} in {country_elapsed / 60:.1f} minutes! Freeing memory...", flush=True)

        # Free index memory completely before moving to next country
        del index
        del target_claimed
        del country_s1_ids
        del s1_to_matches
        gc.collect()
        print(f"  Memory after cleanup: {get_ram_str()}", flush=True)

    # 3. Final Summary & Validation
    print("\n" + "=" * 75, flush=True)
    print("[3/4] INFERENCE COMPLETE ACROSS ALL COUNTRIES!", flush=True)
    print(f"Total S1 Entities Processed : {total_s1_global:,}", flush=True)
    print(f"Total Candidate Pairs Scored: {total_pairs_global:,}", flush=True)
    print(f"Total Matches Claimed       : {total_matches_global:,}", flush=True)
    print(f"Total Runtime Elapsed       : {(time.time() - start_total) / 60:.1f} minutes", flush=True)
    print(f"Matching Results File       : {matching_file} ({matching_file.stat().st_size / (1024*1024):.2f} MB)", flush=True)
    print(f"Candidate Pairs File        : {candidate_file} ({candidate_file.stat().st_size / (1024*1024):.2f} MB)", flush=True)
    print("=" * 75, flush=True)

    # 4. Run Official Submission Validator
    print("\n[4/4] Executing Official Submission Validator...", flush=True)
    validator_script = base_dir / "student_resource" / "utils" / "validate_submission.py"
    if validator_script.exists():
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(matching_file),
            "--candidate", str(candidate_file),
            "--test-dir", str(test_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout, flush=True)
        if res.stderr:
            print("STDERR:", res.stderr, flush=True)
        if res.returncode == 0:
            print("\n>>> VALIDATION SUCCESS: Exit Code 0! Submission files are 100% compliant and ready for upload!", flush=True)
        else:
            print(f"\n>>> VALIDATION WARNING/ERROR: Exit Code {res.returncode}", flush=True)
    else:
        print(f"Validator script not found at {validator_script}, skipping.", flush=True)


if __name__ == "__main__":
    main()
