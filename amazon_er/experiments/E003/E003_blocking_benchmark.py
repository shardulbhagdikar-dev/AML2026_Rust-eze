"""
E003: Ultra-Efficient Enhanced Multi-Signal Blocking Benchmark
Amazon ML Challenge 2026 - Business Entity Resolution

Uses 32-bit unsigned integer arrays (array('I')) for posting lists:
Consumes only 4 bytes per posting instead of ~80 bytes per Python string object,
running completely within a tiny memory footprint (< 1 GB RAM).

Evaluates high-recall blocking signals:
1. Core Name (legal suffix & domain stripped)
2. Compressed Name (compact alphanumeric for domain / no-space matches)
3. Rare Name Word Pairs (e.g. ('trinity', 'lutheran'))
4. Address Signatures (country, number, street_word)
5. Fallback Rare Single Token
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from array import array
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Stopwords & common business terms to never index as single tokens
COMMON_TERMS = {
    "and", "the", "of", "in", "for", "at", "to", "a", "an",
    "pvt", "ltd", "private", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "llp", "co", "company",
    "enterprises", "enterprise", "services", "service", "solutions", "solution",
    "industries", "industry", "trading", "group", "holdings", "holding",
    "international", "india", "usa", "us", "store", "stores",
    "center", "centre", "shop", "com", "net", "org", "hotel", "care"
}

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b(pvt\s*ltd|private\s*limited|ltd|limited|inc|incorporated|llc|llp|corp|corporation|co|company|enterprises|enterprise|solutions|services|group)\b",
    re.IGNORECASE,
)

DOMAIN_REGEX = re.compile(r"(\.com|\.net|\.org|\.in|\.co|\.us|#\d+)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
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
    text = str(name_raw).casefold()
    text = DOMAIN_REGEX.sub("", text)
    text = LEGAL_SUFFIX_REGEX.sub("", text)
    return re.sub(r"[^a-z0-9]", "", text)


def extract_address_signatures(country: str, address_norm: str) -> list[tuple[str, str, str]]:
    if not address_norm:
        return []

    tokens = address_norm.split()
    numbers = [t.lstrip("0") for t in tokens if t.isdigit() and len(t) <= 7]
    numbers = [n for n in numbers if n]

    addr_stopwords = {"street", "road", "lane", "avenue", "drive", "suite", "floor", "near", "opposite", "nagar", "block", "sector", "post", "dist", "state", "city"}
    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in addr_stopwords]

    signatures = []
    c = country.casefold()
    for num in numbers[:2]:
        for w in words[:3]:
            signatures.append((c, num, w))

    return signatures


def extract_name_pairs(name_norm: str) -> list[tuple[str, str]]:
    words = [w for w in name_norm.split() if len(w) >= 3 and w not in COMMON_TERMS]
    if len(words) < 2:
        return []
    pairs = []
    for i in range(len(words) - 1):
        w1, w2 = sorted([words[i], words[i+1]])
        pairs.append((w1, w2))
    return pairs


def extract_rare_tokens(name_norm: str) -> list[str]:
    return [w for w in name_norm.split() if len(w) >= 5 and w not in COMMON_TERMS]


class CompactBlockingIndex:
    """Uses array('I') for 4-byte uint32 postings."""
    def __init__(self, max_postings: int = 150):
        self.max_postings = max_postings
        self.target_ids: list[str] = []

        self.exact_name_idx = defaultdict(lambda: array("I"))
        self.core_name_idx = defaultdict(lambda: array("I"))
        self.compressed_name_idx = defaultdict(lambda: array("I"))
        self.name_pair_idx = defaultdict(lambda: array("I"))
        self.addr_sig_idx = defaultdict(lambda: array("I"))
        self.rare_token_idx = defaultdict(lambda: array("I"))

    def add_target(self, target_id: str, country: str, name_raw: str, addr_raw: str):
        target_idx = len(self.target_ids)
        self.target_ids.append(target_id)

        c = country.casefold()
        name_norm = normalize_text(name_raw)
        addr_norm = normalize_text(addr_raw)

        # 1. Exact Name
        if name_norm:
            k = (c, name_norm)
            lst = self.exact_name_idx[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 2. Core Name
        core = extract_core_name(name_norm)
        if core and len(core) >= 3:
            k = (c, core)
            lst = self.core_name_idx[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 3. Compressed Name
        comp = extract_compressed_name(name_raw)
        if comp and len(comp) >= 5:
            k = (c, comp)
            lst = self.compressed_name_idx[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 4. Name Word Pairs
        for w1, w2 in extract_name_pairs(name_norm):
            k = (c, w1, w2)
            lst = self.name_pair_idx[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 5. Address Signatures
        for sig in extract_address_signatures(c, addr_norm):
            lst = self.addr_sig_idx[sig]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 6. Rare Tokens
        for tok in extract_rare_tokens(name_norm):
            k = (c, tok)
            lst = self.rare_token_idx[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

    def retrieve(self, country: str, name_raw: str, addr_raw: str, max_candidates: int = 80) -> set[str]:
        c = country.casefold()
        name_norm = normalize_text(name_raw)
        addr_norm = normalize_text(addr_raw)

        cand_indices: set[int] = set()

        # Priority 1: Exact Name, Core Name, Compressed Name
        if name_norm and (c, name_norm) in self.exact_name_idx:
            cand_indices.update(self.exact_name_idx[(c, name_norm)])

        core = extract_core_name(name_norm)
        if core and (c, core) in self.core_name_idx:
            cand_indices.update(self.core_name_idx[(c, core)])

        comp = extract_compressed_name(name_raw)
        if comp and (c, comp) in self.compressed_name_idx:
            cand_indices.update(self.compressed_name_idx[(c, comp)])

        # Priority 2: Address Signatures (Number + Street Word)
        for sig in extract_address_signatures(c, addr_norm):
            if sig in self.addr_sig_idx:
                cand_indices.update(self.addr_sig_idx[sig])

        # Priority 3: Name Word Pairs
        for w1, w2 in extract_name_pairs(name_norm):
            k = (c, w1, w2)
            if k in self.name_pair_idx:
                cand_indices.update(self.name_pair_idx[k])

        # Priority 4: Rare single tokens (fallback if candidates are few)
        if len(cand_indices) < 20:
            for tok in extract_rare_tokens(name_norm):
                k = (c, tok)
                if k in self.rare_token_idx:
                    lst = self.rare_token_idx[k]
                    if len(lst) <= 50:
                        cand_indices.update(lst)

        if len(cand_indices) > max_candidates:
            # Truncate
            cand_indices = set(list(cand_indices)[:max_candidates])

        target_ids = self.target_ids
        return {target_ids[idx] for idx in cand_indices}


def main():
    print("=" * 70)
    print("E003: Compact Multi-Signal Blocking Benchmark (Low-Memory array['I'])")
    print("=" * 70)

    data_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource\dataset\train")
    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"
    gt_path = data_dir / "train_ground_truth.tsv"

    print("Loading Ground Truth...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    ground_truth = {}
    for row in gt_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched = str(row.matched_entity_ids).strip()
        ground_truth[s1_id] = {x.strip() for x in matched.split(",") if x.strip()} if matched else set()

    print("Loading S1 and selecting 10,000 benchmark sample (seed=42)...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s1_sample = s1_df.sample(n=10000, random_state=42).reset_index(drop=True)

    sample_s1_ids = set(s1_sample["entity_id"])
    total_true_matches = sum(len(ground_truth[sid]) for sid in sample_s1_ids if sid in ground_truth)
    print(f"Sample S1 count: {len(s1_sample):,}")
    print(f"Total true matches in sample: {total_true_matches:,}")

    index = CompactBlockingIndex(max_postings=120)

    # Stream S2
    print("\nIndexing Source 2 (5,034,616 records)...")
    t0 = time.time()
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            index.add_target(row.entity_id, row.country, row.business_name, row.business_address)
    print(f"  S2 indexed in {time.time() - t0:.1f}s | Targets stored: {len(index.target_ids):,}")

    # Stream S3
    print("\nIndexing Source 3 (5,285,603 records)...")
    t0 = time.time()
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            index.add_target(row.entity_id, row.country, row.business_name, row.business_address)
    print(f"  S3 indexed in {time.time() - t0:.1f}s | Total targets stored: {len(index.target_ids):,}")

    print("\nEvaluating Candidate Retrieval on 10,000 S1 Entities...")
    t0 = time.time()
    matches_found = 0
    total_cands = 0
    candidate_counts = []
    zero_cands = 0

    for row in s1_sample.itertuples(index=False):
        s1_id = row.entity_id
        true_set = ground_truth.get(s1_id, set())

        cands = index.retrieve(row.country, row.business_name, row.business_address, max_candidates=80)
        c_len = len(cands)
        total_cands += c_len
        candidate_counts.append(c_len)

        if c_len == 0:
            zero_cands += 1

        matches_found += len(true_set & cands)

    elapsed = time.time() - t0
    recall = matches_found / total_true_matches if total_true_matches > 0 else 0
    mean_cands = total_cands / len(s1_sample)

    print("\n" + "=" * 70)
    print("E003 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Candidate Recall          : {recall * 100:.2f}% ({matches_found:,} / {total_true_matches:,})")
    print(f"Total Candidates Generated: {total_cands:,}")
    print(f"Mean Candidates per S1    : {mean_cands:.1f}")
    print(f"Median Candidates per S1  : {sorted(candidate_counts)[len(candidate_counts)//2]}")
    print(f"P95 Candidates per S1     : {sorted(candidate_counts)[int(len(candidate_counts)*0.95)]}")
    print(f"Max Candidates per S1     : {max(candidate_counts)}")
    print(f"Zero-candidate S1 rate    : {zero_cands / len(s1_sample) * 100:.2f}% ({zero_cands} entities)")
    print(f"Retrieval Speed           : {len(s1_sample) / elapsed:.0f} entities/sec ({elapsed:.1f}s total)")
    print("=" * 70)

    # Comparison against E002
    print("\nComparison against E002 Combined:")
    print("  E002 Recall   : 61.83%  | Candidates/S1: 2,279.2")
    print(f"  E003 Recall   : {recall * 100:.2f}%  | Candidates/S1: {mean_cands:.1f}")
    diff_recall = (recall - 0.618327) * 100
    print(f"  Recall Change : {diff_recall:+.2f}%")
    print(f"  Candidate Reduction: {(1 - mean_cands / 2279.1767) * 100:.1f}% reduction in pair volume!")


if __name__ == "__main__":
    main()
