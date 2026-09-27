#!/usr/bin/env python3
"""
E007: High-Recall Candidate Generation Engine (Targeting >= 95% Recall)

Key Enhancements over E004.5:
1. Regex Digit Extraction: re.findall(r'\\d+', addr) captures all building/floor/plot numbers
   (e.g., 12Thfloor/1 -> 12, 1; A-301 -> 301; 40/445 -> 40, 445; No.8 -> 8).
2. Cross-Script Transliteration via anyascii:
   Converts Hindi, Tamil, Telugu, Gujarati, Bengali, and French diacritics into ASCII,
   turning cross-script variants into exact/prefix matches.
3. Distinctive Single Core Words:
   Index unique long core words (len >= 6) with posting limits up to 150.
4. Postal Anchor expansion:
   Support 5-digit and 6-digit postal anchors with regex number extraction.

Run command:
  C:\\Users\\Shardul\\AppData\\Local\\Python\\bin\\python.exe amazon_er/experiments/E007/test_e007_recall.py --sample-size 10000
"""

import argparse
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

# Paths
base_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource")
data_dir = base_dir / "dataset" / "train"
s1_path = data_dir / "train_source1.tsv"
s2_path = data_dir / "train_source2.tsv"
s3_path = data_dir / "train_source3.tsv"
gt_path = data_dir / "train_ground_truth.tsv"

out_dir = base_dir / "student_resource" / "amazon_er" / "outputs" / "E007_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Regex & Normalization
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
    "post", "po", "via", "circle", "layout", "extension", "ext", "stage", "rue"
}

DIGIT_REGEX = re.compile(r"\d+")


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
    text = anyascii.anyascii(name_raw).casefold()
    text = DOMAIN_REGEX.sub("", text)
    text = LEGAL_SUFFIX_REGEX.sub("", text)
    return re.sub(r"[^a-z0-9]", "", text)


def extract_all_numbers(address_norm: str) -> list[str]:
    """Extracts all distinct digit sequences from address, stripping leading zeros."""
    matches = DIGIT_REGEX.findall(address_norm)
    res = []
    seen = set()
    for m in matches:
        num = m.lstrip("0")
        if num and len(num) <= 7 and num not in seen:
            seen.add(num)
            res.append(num)
    return res


def extract_enhanced_address_signatures(country: str, address_norm: str) -> list[tuple[str, str, str]]:
    if not address_norm:
        return []
    c = country.casefold()
    tokens = address_norm.split()
    numbers = extract_all_numbers(address_norm)
    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS]

    signatures = []
    for num in numbers[:4]:
        is_small = len(num) < 3
        valid_words = [w for w in words if len(w) >= 5] if is_small else words
        for w in valid_words[:4]:
            signatures.append((c, num, w))
    return signatures


def extract_enhanced_postal_anchors(country: str, address_norm: str) -> list[tuple[str, str, str]]:
    if not address_norm:
        return []
    c = country.casefold()
    tokens = address_norm.split()
    anchors = []

    # Regex search for postal codes
    if c == "in":
        codes = [t for t in DIGIT_REGEX.findall(address_norm) if len(t) == 6 and t[0] in "123456789"]
    elif c in ("us", "fr"):
        codes = [t for t in DIGIT_REGEX.findall(address_norm) if len(t) == 5]
    else:
        codes = [t for t in DIGIT_REGEX.findall(address_norm) if len(t) in (5, 6)]

    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS]
    for pc in codes[:2]:
        for w in words[:3]:
            anchors.append((c, pc, w))
    return anchors


def extract_name_pairs(name_norm: str) -> list[tuple[str, str]]:
    tokens = [t for t in name_norm.split() if len(t) >= 3 and t not in ADDRESS_GENERIC_WORDS]
    if len(tokens) >= 2:
        pairs = []
        for i in range(min(5, len(tokens))):
            for j in range(i + 1, min(5, len(tokens))):
                pairs.append((tokens[i], tokens[j]))
        return pairs
    return []


def extract_prefix_pairs(name_norm: str) -> list[tuple[str, str]]:
    tokens = [t for t in name_norm.split() if len(t) >= 3]
    if len(tokens) >= 2:
        return [(tokens[0][:4], tokens[1][:4])]
    return []


# ----------------------------------------------------------------------
# E007 High-Recall Inverted Index
# ----------------------------------------------------------------------

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


class E007Index:
    def __init__(self, max_postings: int = 150):
        self.max_postings = max_postings
        self.target_ids: list[str] = []

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

    def add_target(self, target_id: str, country: str, name_raw: str, addr_raw: str):
        target_idx = len(self.target_ids)
        self.target_ids.append(target_id)

        c = country.casefold()
        name_norm = normalize_text(name_raw)
        addr_norm = normalize_text(addr_raw)

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

            # Single distinctive core word
            core_words = [w for w in core.split() if len(w) >= 6]
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

        # 6. Name Pairs (including transliterated)
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

        # 8. Address Signatures (Regex digit extracted)
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

    def retrieve(self, country: str, name_raw: str, addr_raw: str) -> dict[int, float]:
        """Returns dict of target_idx -> candidate score"""
        c = country.casefold()
        name_norm = normalize_text(name_raw)
        addr_norm = normalize_text(addr_raw)
        trans_name = transliterate_text(name_raw)
        trans_addr = transliterate_text(addr_raw)

        cand_scores = defaultdict(float)

        # Exact Name
        if name_norm and (c, name_norm) in self.idx_exact_name:
            w = SIGNAL_WEIGHTS["exact_name"]
            for idx in self.idx_exact_name[(c, name_norm)]:
                cand_scores[idx] += w

        # Transliterated Exact
        if trans_name and (c, trans_name) in self.idx_translit_exact:
            w = SIGNAL_WEIGHTS["translit_exact"]
            for idx in self.idx_translit_exact[(c, trans_name)]:
                cand_scores[idx] += w

        # Core Name
        core = extract_core_name(name_norm)
        if core and (c, core) in self.idx_core_name:
            w = SIGNAL_WEIGHTS["core_name"]
            for idx in self.idx_core_name[(c, core)]:
                cand_scores[idx] += w

        # Transliterated Core
        trans_core = extract_core_name(trans_name)
        if trans_core and (c, trans_core) in self.idx_translit_core:
            w = SIGNAL_WEIGHTS["translit_core"]
            for idx in self.idx_translit_core[(c, trans_core)]:
                cand_scores[idx] += w

        # Single Core Word
        if core and (c, core) in self.idx_single_core:
            w = SIGNAL_WEIGHTS["single_core_word"]
            for idx in self.idx_single_core[(c, core)]:
                cand_scores[idx] += w

        # Compressed Name
        comp = extract_compressed_name(name_raw)
        if comp and (c, comp) in self.idx_compressed_name:
            w = SIGNAL_WEIGHTS["compressed_name"]
            for idx in self.idx_compressed_name[(c, comp)]:
                cand_scores[idx] += w

        # Name Pairs
        for w1, w2 in extract_name_pairs(trans_name or name_norm):
            k = (c, w1, w2)
            if k in self.idx_name_pairs:
                w = SIGNAL_WEIGHTS["name_pairs"]
                for idx in self.idx_name_pairs[k]:
                    cand_scores[idx] += w

        # Address Signatures
        for sig in extract_enhanced_address_signatures(c, trans_addr or addr_norm):
            if sig in self.idx_addr_sig:
                w = SIGNAL_WEIGHTS["addr_sig"]
                for idx in self.idx_addr_sig[sig]:
                    cand_scores[idx] += w

        # Postal Anchors
        for anchor in extract_enhanced_postal_anchors(c, trans_addr or addr_norm):
            if anchor in self.idx_postal:
                w = SIGNAL_WEIGHTS["postal_street"]
                for idx in self.idx_postal[anchor]:
                    cand_scores[idx] += w

        # Fallbacks if low candidates
        if len(cand_scores) < 30:
            for p1, p2 in extract_prefix_pairs(trans_name or name_norm):
                k = (c, p1, p2)
                if k in self.idx_prefix_pairs:
                    lst = self.idx_prefix_pairs[k]
                    if len(lst) <= 80:
                        w = SIGNAL_WEIGHTS["prefix_pairs"]
                        for idx in lst:
                            cand_scores[idx] += w

        if len(cand_scores) < 20:
            tokens = [t for t in (trans_name or name_norm).split() if len(t) >= 6 and not t.isdigit()]
            for tok in tokens[:2]:
                k = (c, tok)
                if k in self.idx_rare_tokens:
                    lst = self.idx_rare_tokens[k]
                    if len(lst) <= 50:
                        w = SIGNAL_WEIGHTS["rare_tokens"]
                        for idx in lst:
                            cand_scores[idx] += w

        return cand_scores


def main():
    parser = argparse.ArgumentParser(description="Test E007 High-Recall Candidate Generation")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of S1 entities to benchmark")
    args = parser.parse_args()

    print("=" * 75)
    print(f"E007: HIGH-RECALL CANDIDATE GENERATION BENCHMARK ({args.sample_size:,} S1 Entities)")
    print("=" * 75)

    # 1. Load Ground Truth
    print("\n[1/4] Loading Ground Truth...")
    gt = defaultdict(set)
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
        for sid, m in zip(chunk["source1_entity_id"].astype(str), chunk["matched_entity_ids"].astype(str)):
            m_str = m.strip()
            if m_str:
                gt[sid] = {x.strip() for x in m_str.split(",") if x.strip()}

    # 2. Sample S1
    print(f"\n[2/4] Sampling {args.sample_size:,} S1 entities...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s1_sample = s1_df.sample(n=min(args.sample_size, len(s1_df)), random_state=42).reset_index(drop=True)
    sample_s1_ids = set(s1_sample["entity_id"])
    total_true = sum(len(gt[sid]) for sid in sample_s1_ids if sid in gt)
    print(f"  Total true matches in sample: {total_true:,}")

    # 3. Index Targets
    index = E007Index(max_postings=150)
    print("\n[3/4] Indexing Source 2 & Source 3 with Transliteration & Regex Numbers...")
    t0 = time.time()
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
        for eid, c, n, a in zip(chunk["entity_id"].astype(str), chunk["country"].astype(str), chunk["business_name"].astype(str), chunk["business_address"].astype(str)):
            index.add_target(eid, c, n, a)
    print(f"  S2 indexed in {time.time() - t0:.1f}s | Targets: {len(index.target_ids):,}")

    t0 = time.time()
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
        for eid, c, n, a in zip(chunk["entity_id"].astype(str), chunk["country"].astype(str), chunk["business_name"].astype(str), chunk["business_address"].astype(str)):
            index.add_target(eid, c, n, a)
    print(f"  S3 indexed in {time.time() - t0:.1f}s | Total Targets: {len(index.target_ids):,}")

    # 4. Candidate Retrieval Benchmark across different caps
    print(f"\n[4/4] Querying {len(s1_sample):,} S1 entities...")
    caps = [60, 80, 100, 120, 150]
    cap_found = {c: 0 for c in caps}
    untruncated_found = 0
    total_candidates_by_cap = {c: 0 for c in caps}

    s1_ids = s1_sample["entity_id"].astype(str).tolist()
    s1_countries = s1_sample["country"].astype(str).tolist()
    s1_names = s1_sample["business_name"].astype(str).tolist()
    s1_addrs = s1_sample["business_address"].astype(str).tolist()

    t_ret = time.time()
    target_ids = index.target_ids

    for i, (s1_id, country, name, addr) in enumerate(zip(s1_ids, s1_countries, s1_names, s1_addrs), start=1):
        true_set = gt.get(s1_id, set())

        cand_scores = index.retrieve(country, name, addr)
        sorted_targets = sorted(cand_scores.keys(), key=lambda idx: cand_scores[idx], reverse=True)

        # Check untruncated
        found_in_entity = set()
        for idx in sorted_targets:
            tid = target_ids[idx]
            if tid in true_set:
                found_in_entity.add(tid)
        untruncated_found += len(found_in_entity)

        # Check caps
        for cap in caps:
            top_k = sorted_targets[:cap]
            found_k = sum(1 for idx in top_k if target_ids[idx] in true_set)
            cap_found[cap] += found_k
            total_candidates_by_cap[cap] += len(top_k)

        if i % 2500 == 0:
            elapsed = time.time() - t_ret
            print(f"  Progress: {i:,}/{len(s1_sample):,} | Rate: {i/elapsed:.0f} ent/s")

    elapsed_total = time.time() - t_ret
    untruncated_recall = (untruncated_found / total_true) * 100 if total_true else 0

    print("\n" + "=" * 75)
    print("E007 RECALL BENCHMARK RESULTS")
    print("=" * 75)
    print(f"Total True Matches Evaluated : {total_true:,}")
    print(f"Untruncated Recall Ceiling   : {untruncated_recall:.2f}% ({untruncated_found:,} / {total_true:,})")
    print(f"Query Throughput Rate        : {len(s1_sample) / elapsed_total:.0f} entities/sec")
    print("-" * 75)
    print(f"{'Candidate Cap':<15} | {'True Found':<12} | {'Recall (%)':<12} | {'Mean Cands/S1':<15}")
    print("-" * 75)

    results_table = []
    for cap in caps:
        rec = (cap_found[cap] / total_true) * 100
        mean_cands = total_candidates_by_cap[cap] / len(s1_sample)
        print(f"Top {cap:<11} | {cap_found[cap]:<12,} | {rec:<12.2f}% | {mean_cands:<15.1f}")
        results_table.append({
            "cap": cap,
            "true_found": cap_found[cap],
            "recall": round(rec, 4),
            "mean_candidates": round(mean_cands, 1),
        })
    print("=" * 75)

    payload = {
        "experiment": "E007",
        "sample_size": args.sample_size,
        "total_true_matches": total_true,
        "untruncated_recall": untruncated_recall,
        "caps": results_table,
    }
    with open(out_dir / "e007_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Metrics saved to {out_dir / 'e007_metrics.json'}")


if __name__ == "__main__":
    main()
