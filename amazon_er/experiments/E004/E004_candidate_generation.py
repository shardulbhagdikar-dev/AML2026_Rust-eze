"""
E004: Enhanced Multi-Signal Candidate Generation with Signal Attribution
Amazon ML Challenge 2026 - Business Entity Resolution

Key Innovations over E003:
1. Signal Attribution: Measures individual & cumulative recall for each signal.
2. Clean Address Signatures: Filters generic single/double-digit noise, normalizes leading zeros (001023 -> 1023).
3. Expanded Core Name Normalization: Strips comprehensive legal/business suffixes and domain artifacts.
4. Name Prefix Pairs & Distinctive N-gram Fingerprints: Resilient to typos (e.g., Chrch vs Church).
5. Per-Signal Quotas: Prevents generic address matches from crowding out high-confidence name candidates.
6. Memory-Safe 4-byte uint32 posting lists via array('I').
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

# Comprehensive legal suffixes & business terms
LEGAL_TERMS = {
    "pvt ltd", "private limited", "pvt limited", "private ltd", "limited", "ltd",
    "inc", "incorporated", "llc", "llp", "corp", "corporation", "co", "company",
    "enterprises", "enterprise", "services", "service", "solutions", "solution",
    "industries", "industry", "trading", "group", "holdings", "holding",
    "international", "consultants", "consultancy", "associates", "store", "stores",
    "center", "centre", "shop"
}

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b(pvt\s*ltd|private\s*limited|pvt\s*limited|private\s*ltd|limited|ltd|"
    r"inc|incorporated|llc|llp|corp|corporation|co|company|enterprises|enterprise|"
    r"services|service|solutions|solution|industries|industry|trading|group|"
    r"holdings|holding|international|consultants|consultancy|associates)\b",
    re.IGNORECASE,
)

DOMAIN_REGEX = re.compile(
    r"(\.com|\.net|\.org|\.in|\.co\.in|\.co|\.us|\.io|\.biz|\.info|#\d+|www\.)",
    re.IGNORECASE,
)

# Common words that must never act as standalone single blocking keys
STOPWORDS = {
    "and", "the", "of", "in", "for", "at", "to", "a", "an", "on", "by", "with",
    "pvt", "ltd", "private", "limited", "inc", "corp", "llc", "co", "company",
    "services", "solutions", "enterprises", "trading", "group", "holdings",
    "india", "usa", "us", "store", "center", "shop", "hotel", "care"
}

# Generic address terms that cannot establish a distinctive signature
ADDRESS_GENERIC_WORDS = {
    "street", "st", "road", "rd", "lane", "ln", "avenue", "ave", "drive", "dr",
    "suite", "ste", "floor", "fl", "building", "bldg", "near", "opposite", "opp",
    "nagar", "block", "blk", "sector", "sec", "post", "dist", "state", "city",
    "north", "south", "east", "west", "main", "cross", "highway", "hwy", "way",
    "park", "plaza", "room", "apartment", "apt", "unit", "box", "po"
}


def normalize_text(text: str) -> str:
    """Standard unicode normalization, lowercasing, and whitespace collapse."""
    if not text or pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def extract_core_name(name_norm: str) -> str:
    """Removes legal suffixes and domain artifacts to isolate the entity brand."""
    if not name_norm:
        return ""
    cleaned = DOMAIN_REGEX.sub("", name_norm)
    cleaned = LEGAL_SUFFIX_REGEX.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_compressed_name(name_raw: str) -> str:
    """Removes non-alphanumeric chars and extensions for domain/space-free matching."""
    if not name_raw or pd.isna(name_raw):
        return ""
    text = str(name_raw).casefold()
    text = DOMAIN_REGEX.sub("", text)
    text = LEGAL_SUFFIX_REGEX.sub("", text)
    return re.sub(r"[^a-z0-9]", "", text)


def extract_clean_address_signatures(country: str, address_norm: str) -> list[tuple[str, str, str]]:
    """
    Extracts high-precision address signatures: (country, normalized_number, distinctive_street_word).
    Filters out generic low-number noise and strips leading zeros.
    """
    if not address_norm:
        return []

    tokens = address_norm.split()

    # Normalize numbers: strip leading zeros ('001023' -> '1023')
    raw_numbers = [t.lstrip("0") for t in tokens if t.isdigit()]
    numbers = [n for n in raw_numbers if n and len(n) <= 7]

    # Distinctive words: length >= 4 and not generic street/floor words
    words = [
        t for t in tokens
        if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS
    ]

    signatures = []
    c = country.casefold()

    for num in numbers[:3]:
        # Filter: if number < 100, require a longer, non-generic word (>= 5 chars)
        is_small_num = len(num) < 3
        valid_words = [w for w in words if len(w) >= 5] if is_small_num else words

        for w in valid_words[:2]:
            signatures.append((c, num, w))

    return signatures


def extract_name_pairs(name_norm: str) -> list[tuple[str, str]]:
    """Extracts sorted adjacent pairs of distinctive words."""
    words = [w for w in name_norm.split() if len(w) >= 3 and w not in STOPWORDS]
    if len(words) < 2:
        return []
    pairs = []
    for i in range(len(words) - 1):
        w1, w2 = sorted([words[i], words[i+1]])
        pairs.append((w1, w2))
    return pairs


def extract_prefix_pairs(name_norm: str) -> list[tuple[str, str]]:
    """
    Extracts 4-character prefix pairs to absorb typos at word boundaries.
    Example: 'trinity lutheran' -> ('trin', 'luth')
    """
    words = [w[:4] for w in name_norm.split() if len(w) >= 4 and w not in STOPWORDS]
    if len(words) < 2:
        return []
    pairs = []
    for i in range(len(words) - 1):
        p1, p2 = sorted([words[i], words[i+1]])
        pairs.append((p1, p2))
    return pairs


def extract_distinctive_tokens(name_norm: str) -> list[str]:
    """Extracts rare or long tokens that are informative by themselves."""
    return [w for w in name_norm.split() if len(w) >= 6 and w not in STOPWORDS]


class E004BlockingIndex:
    """
    Multi-index candidate generation system with 4-byte uint32 postings.
    """
    def __init__(self, max_postings: int = 150):
        self.max_postings = max_postings
        self.target_ids: list[str] = []

        self.idx_exact_name = defaultdict(lambda: array("I"))
        self.idx_core_name = defaultdict(lambda: array("I"))
        self.idx_compressed_name = defaultdict(lambda: array("I"))
        self.idx_name_pairs = defaultdict(lambda: array("I"))
        self.idx_prefix_pairs = defaultdict(lambda: array("I"))
        self.idx_addr_sig = defaultdict(lambda: array("I"))
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

        # 2. Core Name
        core = extract_core_name(name_norm)
        if core and len(core) >= 3:
            k = (c, core)
            lst = self.idx_core_name[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 3. Compressed Name
        comp = extract_compressed_name(name_raw)
        if comp and len(comp) >= 5:
            k = (c, comp)
            lst = self.idx_compressed_name[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 4. Word Pairs
        for w1, w2 in extract_name_pairs(name_norm):
            k = (c, w1, w2)
            lst = self.idx_name_pairs[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 5. Prefix Pairs
        for p1, p2 in extract_prefix_pairs(name_norm):
            k = (c, p1, p2)
            lst = self.idx_prefix_pairs[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 6. Address Signatures
        for sig in extract_clean_address_signatures(c, addr_norm):
            lst = self.idx_addr_sig[sig]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 7. Rare Tokens
        for tok in extract_distinctive_tokens(name_norm):
            k = (c, tok)
            lst = self.idx_rare_tokens[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

    def retrieve_with_attribution(
        self,
        country: str,
        name_raw: str,
        addr_raw: str,
        max_total: int = 100,
    ) -> tuple[set[str], dict[str, set[str]]]:
        """
        Retrieves candidates and attributes which signals found which candidate.
        """
        c = country.casefold()
        name_norm = normalize_text(name_raw)
        addr_norm = normalize_text(addr_raw)

        signal_cands: dict[str, set[int]] = {
            "exact_name": set(),
            "core_name": set(),
            "compressed_name": set(),
            "name_pairs": set(),
            "prefix_pairs": set(),
            "addr_sig": set(),
            "rare_tokens": set(),
        }

        # 1. Exact Name
        if name_norm and (c, name_norm) in self.idx_exact_name:
            signal_cands["exact_name"].update(self.idx_exact_name[(c, name_norm)])

        # 2. Core Name
        core = extract_core_name(name_norm)
        if core and (c, core) in self.idx_core_name:
            signal_cands["core_name"].update(self.idx_core_name[(c, core)])

        # 3. Compressed Name
        comp = extract_compressed_name(name_raw)
        if comp and (c, comp) in self.idx_compressed_name:
            signal_cands["compressed_name"].update(self.idx_compressed_name[(c, comp)])

        # 4. Word Pairs
        for w1, w2 in extract_name_pairs(name_norm):
            k = (c, w1, w2)
            if k in self.idx_name_pairs:
                signal_cands["name_pairs"].update(self.idx_name_pairs[k])

        # 5. Clean Address Signatures
        for sig in extract_clean_address_signatures(c, addr_norm):
            if sig in self.idx_addr_sig:
                signal_cands["addr_sig"].update(self.idx_addr_sig[sig])

        # 6. Prefix Pairs (only if name candidates are moderate)
        if len(signal_cands["name_pairs"]) < 30:
            for p1, p2 in extract_prefix_pairs(name_norm):
                k = (c, p1, p2)
                if k in self.idx_prefix_pairs:
                    lst = self.idx_prefix_pairs[k]
                    if len(lst) <= 60:
                        signal_cands["prefix_pairs"].update(lst)

        # 7. Distinctive Tokens (fallback)
        total_so_far = sum(len(s) for s in signal_cands.values())
        if total_so_far < 15:
            for tok in extract_distinctive_tokens(name_norm):
                k = (c, tok)
                if k in self.idx_rare_tokens:
                    lst = self.idx_rare_tokens[k]
                    if len(lst) <= 40:
                        signal_cands["rare_tokens"].update(lst)

        # Combine with quota priority:
        # High confidence name candidates come first, followed by address, then fuzzy
        combined_indices: list[int] = []

        priority_order = [
            "exact_name", "core_name", "compressed_name",
            "name_pairs", "addr_sig", "prefix_pairs", "rare_tokens"
        ]

        seen = set()
        for sig in priority_order:
            for idx in signal_cands[sig]:
                if idx not in seen:
                    seen.add(idx)
                    combined_indices.append(idx)
                    if len(combined_indices) >= max_total:
                        break
            if len(combined_indices) >= max_total:
                break

        target_ids = self.target_ids
        final_cands = {target_ids[idx] for idx in combined_indices}

        attributed = {
            sig: {target_ids[idx] for idx in cands}
            for sig, cands in signal_cands.items()
        }

        return final_cands, attributed


def run_e004_benchmark():
    print("=" * 70)
    print("E004: Enhanced Multi-Signal Candidate Generation Benchmark")
    print("=" * 70)

    data_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource\dataset\train")
    output_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource\student_resource\amazon_er\outputs\E004_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print("Selecting 10,000 S1 sample (fixed seed=42)...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s1_sample = s1_df.sample(n=10000, random_state=42).reset_index(drop=True)

    sample_s1_ids = set(s1_sample["entity_id"])
    total_true_matches = sum(len(ground_truth[sid]) for sid in sample_s1_ids if sid in ground_truth)
    print(f"Sample S1 count: {len(s1_sample):,}")
    print(f"Total true matches in sample: {total_true_matches:,}")

    index = E004BlockingIndex(max_postings=120)

    # Index S2
    print("\nIndexing Source 2 (5,034,616 records)...")
    t0 = time.time()
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            index.add_target(row.entity_id, row.country, row.business_name, row.business_address)
    print(f"  S2 indexed in {time.time() - t0:.1f}s | Stored: {len(index.target_ids):,}")

    # Index S3
    print("\nIndexing Source 3 (5,285,603 records)...")
    t0 = time.time()
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            index.add_target(row.entity_id, row.country, row.business_name, row.business_address)
    print(f"  S3 indexed in {time.time() - t0:.1f}s | Total Stored: {len(index.target_ids):,}")

    print("\nEvaluating Candidate Retrieval & Signal Attribution on 10,000 S1 Entities...")
    t0 = time.time()

    signal_hits = defaultdict(int)
    cumulative_hits = 0
    total_cands = 0
    candidate_counts = []
    zero_cands = 0

    cumulative_seen = set()

    for row in s1_sample.itertuples(index=False):
        s1_id = row.entity_id
        true_set = ground_truth.get(s1_id, set())

        final_cands, attributed = index.retrieve_with_attribution(
            row.country, row.business_name, row.business_address, max_total=100
        )

        c_len = len(final_cands)
        total_cands += c_len
        candidate_counts.append(c_len)
        if c_len == 0:
            zero_cands += 1

        # Track cumulative hits in final candidate set
        hits_for_entity = true_set & final_cands
        cumulative_hits += len(hits_for_entity)

        # Track per-signal isolated hits
        for sig, cands in attributed.items():
            signal_hits[sig] += len(true_set & cands)

    elapsed = time.time() - t0
    final_recall = cumulative_hits / total_true_matches if total_true_matches > 0 else 0
    mean_cands = total_cands / len(s1_sample)

    print("\n" + "=" * 70)
    print("E004 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Candidate Recall          : {final_recall * 100:.2f}% ({cumulative_hits:,} / {total_true_matches:,})")
    print(f"Total Candidates Generated: {total_cands:,}")
    print(f"Mean Candidates per S1    : {mean_cands:.1f}")
    print(f"Median Candidates per S1  : {sorted(candidate_counts)[len(candidate_counts)//2]}")
    print(f"P95 Candidates per S1     : {sorted(candidate_counts)[int(len(candidate_counts)*0.95)]}")
    print(f"Max Candidates per S1     : {max(candidate_counts)}")
    print(f"Zero-candidate S1 rate    : {zero_cands / len(s1_sample) * 100:.2f}% ({zero_cands} entities)")
    print(f"Retrieval Speed           : {len(s1_sample) / elapsed:.0f} entities/sec ({elapsed:.1f}s total)")
    print("=" * 70)

    print("\nSignal-by-Signal Isolated Recall Coverage:")
    for sig in ["exact_name", "core_name", "compressed_name", "name_pairs", "addr_sig", "prefix_pairs", "rare_tokens"]:
        h = signal_hits[sig]
        r = (h / total_true_matches) * 100 if total_true_matches else 0
        print(f"  • {sig:<18}: {r:6.2f}% ({h:,} matches found)")

    # Comparison against E002 and E003
    print("\n" + "=" * 70)
    print("PROGRESSION SUMMARY")
    print("=" * 70)
    print(f"  E002 (Baseline Combined): Recall = 61.83% | Candidates/S1 = 2,279.2 (22.8M pairs)")
    print(f"  E003 (Compact Blocking) : Recall = 56.19% | Candidates/S1 =    59.5 (595K pairs)")
    print(f"  E004 (Enhanced Blocking): Recall = {final_recall * 100:.2f}% | Candidates/S1 =    {mean_cands:.1f} ({total_cands:,} pairs)")
    print("=" * 70)

    # Save metrics JSON
    metrics = {
        "experiment": "E004",
        "sample_size": len(s1_sample),
        "total_true_matches": total_true_matches,
        "true_matches_found": cumulative_hits,
        "candidate_recall": final_recall,
        "total_candidates": total_cands,
        "mean_candidates_per_s1": mean_cands,
        "median_candidates_per_s1": sorted(candidate_counts)[len(candidate_counts)//2],
        "zero_candidate_entities": zero_cands,
        "signal_hits": dict(signal_hits),
        "runtime_seconds": elapsed,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    run_e004_benchmark()
