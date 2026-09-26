"""
E004.5: Multi-Signal Weighted Ranking & Postal Anchor Candidate Generation
Amazon ML Challenge 2026 - Business Entity Resolution

Enhancements over E004:
1. Multi-Signal Scoring & Dynamic Ranking:
   Candidates matching multiple independent signals (e.g., core_name + addr_sig)
   are scored and ranked to the top, preventing single-signal distractor flooding.
2. Country-Agnostic Postal Anchor Matching:
   Extracts 4-6 digit postal codes (5-digit France/US, 6-digit India) paired with
   street numbers or core name tokens.
3. Untruncated Recall Ceiling Profiling:
   Measures raw union recall vs Top-80, Top-100, and Top-120 candidate caps.
4. Comprehensive Artifact Generation:
   Saves full terminal logs (E004.5_terminal_op.txt), signal attribution CSV,
   progression comparison CSV, missed-pair diagnostics, and metrics JSON.
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
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Comprehensive legal suffixes & business types
LEGAL_TERMS = {
    "pvt ltd", "private limited", "pvt limited", "private ltd", "pte ltd", "limited", "ltd",
    "inc", "incorporated", "llc", "llp", "corp", "corporation", "co", "company",
    "enterprises", "enterprise", "services", "service", "solutions", "solution",
    "industries", "industry", "trading", "group", "holdings", "holding",
    "international", "consultants", "consultancy", "associates", "store", "stores",
    "center", "centre", "shop", "technologies", "technology", "software"
}

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

# Signal Weights for Multi-Signal Scoring
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


def extract_clean_address_signatures(country: str, address_norm: str) -> list[tuple[str, str, str]]:
    if not address_norm:
        return []

    tokens = address_norm.split()
    raw_numbers = [t.lstrip("0") for t in tokens if t.isdigit()]
    numbers = [n for n in raw_numbers if n and len(n) <= 7]

    words = [
        t for t in tokens
        if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS
    ]

    signatures = []
    c = country.casefold()

    for num in numbers[:3]:
        is_small_num = len(num) < 3
        valid_words = [w for w in words if len(w) >= 5] if is_small_num else words
        for w in valid_words[:2]:
            signatures.append((c, num, w))

    return signatures


def extract_postal_anchors(country: str, address_norm: str) -> list[tuple[str, str, str]]:
    """
    Extracts country-agnostic postal code anchors: (country, postal_code, street_num_or_token).
    Covers France (5 digits), US (5 digits), and India (6 digits).
    """
    if not address_norm:
        return []

    tokens = address_norm.split()
    # Find postal-like numbers (4, 5, or 6 digits)
    postals = [t for t in tokens if t.isdigit() and (4 <= len(t) <= 6)]
    if not postals:
        return []

    # Find secondary street numbers or distinct words
    other_numbers = [t.lstrip("0") for t in tokens if t.isdigit() and t not in postals and t.lstrip("0")]
    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_GENERIC_WORDS]

    anchors = []
    c = country.casefold()
    p = postals[0]  # primary postal code

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


class E0045BlockingIndex:
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

        # 4. Name Word Pairs
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

        # 7. Postal Anchors
        for anchor in extract_postal_anchors(c, addr_norm):
            lst = self.idx_postal[anchor]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

        # 8. Distinctive Tokens
        for tok in extract_distinctive_tokens(name_norm):
            k = (c, tok)
            lst = self.idx_rare_tokens[k]
            if len(lst) < self.max_postings:
                lst.append(target_idx)

    def retrieve(
        self,
        country: str,
        name_raw: str,
        addr_raw: str,
    ) -> tuple[dict[str, set[int]], dict[int, float]]:
        """
        Retrieves candidate indices grouped by signal, and calculates
        multi-signal composite scores for each candidate index.
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
            "postal_street": set(),
            "rare_tokens": set(),
        }

        cand_scores = defaultdict(float)

        # 1. Exact Name
        if name_norm and (c, name_norm) in self.idx_exact_name:
            matches = self.idx_exact_name[(c, name_norm)]
            signal_cands["exact_name"].update(matches)
            w = SIGNAL_WEIGHTS["exact_name"]
            for idx in matches:
                cand_scores[idx] += w

        # 2. Core Name
        core = extract_core_name(name_norm)
        if core and (c, core) in self.idx_core_name:
            matches = self.idx_core_name[(c, core)]
            signal_cands["core_name"].update(matches)
            w = SIGNAL_WEIGHTS["core_name"]
            for idx in matches:
                cand_scores[idx] += w

        # 3. Compressed Name
        comp = extract_compressed_name(name_raw)
        if comp and (c, comp) in self.idx_compressed_name:
            matches = self.idx_compressed_name[(c, comp)]
            signal_cands["compressed_name"].update(matches)
            w = SIGNAL_WEIGHTS["compressed_name"]
            for idx in matches:
                cand_scores[idx] += w

        # 4. Word Pairs
        for w1, w2 in extract_name_pairs(name_norm):
            k = (c, w1, w2)
            if k in self.idx_name_pairs:
                matches = self.idx_name_pairs[k]
                signal_cands["name_pairs"].update(matches)
                w = SIGNAL_WEIGHTS["name_pairs"]
                for idx in matches:
                    cand_scores[idx] += w

        # 5. Address Signatures
        for sig in extract_clean_address_signatures(c, addr_norm):
            if sig in self.idx_addr_sig:
                matches = self.idx_addr_sig[sig]
                signal_cands["addr_sig"].update(matches)
                w = SIGNAL_WEIGHTS["addr_sig"]
                for idx in matches:
                    cand_scores[idx] += w

        # 6. Postal Anchors
        for anchor in extract_postal_anchors(c, addr_norm):
            if anchor in self.idx_postal:
                matches = self.idx_postal[anchor]
                signal_cands["postal_street"].update(matches)
                w = SIGNAL_WEIGHTS["postal_street"]
                for idx in matches:
                    cand_scores[idx] += w

        # 7. Prefix Pairs (if name candidates are few)
        if len(signal_cands["name_pairs"]) < 25:
            for p1, p2 in extract_prefix_pairs(name_norm):
                k = (c, p1, p2)
                if k in self.idx_prefix_pairs:
                    matches = self.idx_prefix_pairs[k]
                    if len(matches) <= 60:
                        signal_cands["prefix_pairs"].update(matches)
                        w = SIGNAL_WEIGHTS["prefix_pairs"]
                        for idx in matches:
                            cand_scores[idx] += w

        # 8. Distinctive Tokens (fallback)
        if len(cand_scores) < 15:
            for tok in extract_distinctive_tokens(name_norm):
                k = (c, tok)
                if k in self.idx_rare_tokens:
                    matches = self.idx_rare_tokens[k]
                    if len(matches) <= 40:
                        signal_cands["rare_tokens"].update(matches)
                        w = SIGNAL_WEIGHTS["rare_tokens"]
                        for idx in matches:
                            cand_scores[idx] += w

        return signal_cands, cand_scores


def main():
    start_all = time.time()
    data_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource\dataset\train")
    output_dir = Path(r"e:\Amazon_ML_Challenge\6ab10eb3b23ba_student_resource\student_resource\amazon_er\outputs\E004.5_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "E004.5_terminal_op.txt"
    log_fp = open(log_file, "w", encoding="utf-8")

    def print_log(msg: str = ""):
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()

    print_log("=" * 70)
    print_log("E004.5: MULTI-SIGNAL WEIGHTED RANKING & POSTAL ANCHORS")
    print_log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_log("=" * 70)

    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"
    gt_path = data_dir / "train_ground_truth.tsv"

    print_log("\nLoading Ground Truth...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    ground_truth = {}
    for row in gt_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched = str(row.matched_entity_ids).strip()
        ground_truth[s1_id] = {x.strip() for x in matched.split(",") if x.strip()} if matched else set()

    print_log("Selecting 10,000 S1 sample (fixed seed=42)...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s1_sample = s1_df.sample(n=10000, random_state=42).reset_index(drop=True)

    sample_s1_ids = set(s1_sample["entity_id"])
    total_true_matches = sum(len(ground_truth[sid]) for sid in sample_s1_ids if sid in ground_truth)
    print_log(f"Sample S1 count: {len(s1_sample):,}")
    print_log(f"Total true matches in sample: {total_true_matches:,}")

    index = E0045BlockingIndex(max_postings=120)

    # Stream S2
    print_log("\nIndexing Source 2 (5,034,616 records)...")
    t0 = time.time()
    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            index.add_target(row.entity_id, row.country, row.business_name, row.business_address)
    print_log(f"  S2 indexed in {time.time() - t0:.1f}s | Stored: {len(index.target_ids):,}")

    # Stream S3
    print_log("\nIndexing Source 3 (5,285,603 records)...")
    t0 = time.time()
    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=500_000, keep_default_na=False):
        for row in chunk.itertuples(index=False):
            index.add_target(row.entity_id, row.country, row.business_name, row.business_address)
    print_log(f"  S3 indexed in {time.time() - t0:.1f}s | Total Stored: {len(index.target_ids):,}")

    print_log("\nEvaluating Candidate Retrieval across multiple capacity tiers...")
    t0 = time.time()

    # Track recalls at different caps: raw untruncated, cap=80, cap=100, cap=120
    hits_untruncated = 0
    hits_cap80 = 0
    hits_cap100 = 0
    hits_cap120 = 0

    cands_cap80_total = 0
    cands_cap100_total = 0
    cands_cap120_total = 0

    signal_hits = defaultdict(int)
    zero_cands_100 = 0
    missed_pairs_sample = []

    target_ids = index.target_ids

    for row in s1_sample.itertuples(index=False):
        s1_id = row.entity_id
        true_set = ground_truth.get(s1_id, set())

        signal_cands, cand_scores = index.retrieve(row.country, row.business_name, row.business_address)

        # Isolated signal hits
        for sig, idx_set in signal_cands.items():
            sig_targets = {target_ids[i] for i in idx_set}
            signal_hits[sig] += len(true_set & sig_targets)

        # Raw untruncated union
        raw_union_targets = {target_ids[i] for i in cand_scores.keys()}
        hits_untruncated += len(true_set & raw_union_targets)

        # Sort candidates by multi-signal score descending
        sorted_indices = sorted(cand_scores.keys(), key=lambda idx: cand_scores[idx], reverse=True)

        # Tier: Cap 80
        cap80_targets = {target_ids[i] for i in sorted_indices[:80]}
        hits_cap80 += len(true_set & cap80_targets)
        cands_cap80_total += len(cap80_targets)

        # Tier: Cap 100
        cap100_targets = {target_ids[i] for i in sorted_indices[:100]}
        h100 = true_set & cap100_targets
        hits_cap100 += len(h100)
        cands_cap100_total += len(cap100_targets)
        if len(cap100_targets) == 0:
            zero_cands_100 += 1

        # Track missed pairs for error analysis
        if true_set and len(missed_pairs_sample) < 50:
            for m in true_set:
                if m not in cap100_targets:
                    missed_pairs_sample.append({
                        "source1_entity_id": s1_id,
                        "missed_target_id": m,
                        "s1_country": row.country,
                        "s1_name": row.business_name,
                        "s1_address": row.business_address,
                    })

        # Tier: Cap 120
        cap120_targets = {target_ids[i] for i in sorted_indices[:120]}
        hits_cap120 += len(true_set & cap120_targets)
        cands_cap120_total += len(cap120_targets)

    eval_time = time.time() - t0
    total_time = time.time() - start_all

    rec_raw = (hits_untruncated / total_true_matches) * 100
    rec_80 = (hits_cap80 / total_true_matches) * 100
    rec_100 = (hits_cap100 / total_true_matches) * 100
    rec_120 = (hits_cap120 / total_true_matches) * 100

    mean_80 = cands_cap80_total / len(s1_sample)
    mean_100 = cands_cap100_total / len(s1_sample)
    mean_120 = cands_cap120_total / len(s1_sample)

    print_log("\n" + "=" * 70)
    print_log("E004.5 BENCHMARK RESULTS")
    print_log("=" * 70)
    print_log(f"Raw Untruncated Recall Ceiling: {rec_raw:.2f}% ({hits_untruncated:,} / {total_true_matches:,})")
    print_log("-" * 70)
    print_log(f"Cap @ 80  Candidates: Recall = {rec_80:6.2f}% ({hits_cap80:,}) | Mean Cands/S1 = {mean_80:.1f}")
    print_log(f"Cap @ 100 Candidates: Recall = {rec_100:6.2f}% ({hits_cap100:,}) | Mean Cands/S1 = {mean_100:.1f}")
    print_log(f"Cap @ 120 Candidates: Recall = {rec_120:6.2f}% ({hits_cap120:,}) | Mean Cands/S1 = {mean_120:.1f}")
    print_log("-" * 70)
    print_log(f"Zero-candidate S1 rate (Cap 100): {zero_cands_100 / len(s1_sample) * 100:.2f}% ({zero_cands_100} entities)")
    print_log(f"Retrieval Speed: {len(s1_sample) / eval_time:.0f} entities/sec ({eval_time:.1f}s)")
    print_log(f"Total Pipeline Time: {total_time:.1f}s")
    print_log("=" * 70)

    print_log("\nSignal-by-Signal Isolated Coverage:")
    sig_rows = []
    for sig in ["exact_name", "core_name", "compressed_name", "name_pairs", "addr_sig", "postal_street", "prefix_pairs", "rare_tokens"]:
        h = signal_hits[sig]
        r = (h / total_true_matches) * 100
        print_log(f"  • {sig:<18}: {r:6.2f}% ({h:,} matches found)")
        sig_rows.append({"signal": sig, "matches_found": h, "recall_percent": round(r, 2)})

    # Save Signal Attribution CSV
    sig_csv_path = output_dir / "E004.5_signal_attribution.csv"
    pd.DataFrame(sig_rows).to_csv(sig_csv_path, index=False)
    print_log(f"\nSignal attribution saved to: {sig_csv_path}")

    # Progression Comparison Table
    print_log("\n" + "=" * 70)
    print_log("PROGRESSION COMPARISON (E002 -> E003 -> E004 -> E004.5)")
    print_log("=" * 70)
    comp_rows = [
        {"experiment": "E002 (Baseline Combined)", "candidate_recall": "61.83%", "mean_candidates": 2279.2, "total_pairs": "22,791,768"},
        {"experiment": "E003 (Compact Blocking)",  "candidate_recall": "56.19%", "mean_candidates": 59.5,   "total_pairs": "595,455"},
        {"experiment": "E004 (Multi-Signal)",      "candidate_recall": "70.89%", "mean_candidates": 67.7,   "total_pairs": "676,710"},
        {"experiment": "E004.5 (Weighted Rank@100)","candidate_recall": f"{rec_100:.2f}%", "mean_candidates": round(mean_100, 1), "total_pairs": f"{cands_cap100_total:,}"},
        {"experiment": "E004.5 (Weighted Rank@120)","candidate_recall": f"{rec_120:.2f}%", "mean_candidates": round(mean_120, 1), "total_pairs": f"{cands_cap120_total:,}"},
    ]
    for row in comp_rows:
        print_log(f"  {row['experiment']:<27}: Recall = {row['candidate_recall']} | Mean Cands = {row['mean_candidates']:>6} | Pairs = {row['total_pairs']}")
    print_log("=" * 70)

    comp_csv_path = output_dir / "E004.5_strategy_comparison.csv"
    pd.DataFrame(comp_rows).to_csv(comp_csv_path, index=False)
    print_log(f"Strategy comparison saved to: {comp_csv_path}")

    # Save missed pairs sample
    if missed_pairs_sample:
        missed_df = pd.DataFrame(missed_pairs_sample)
        missed_path = output_dir / "missed_pairs_sample.csv"
        missed_df.to_csv(missed_path, index=False)
        print_log(f"Saved {len(missed_df)} missed pairs sample to: {missed_path}")

    # Save metrics JSON
    metrics = {
        "experiment": "E004.5",
        "sample_size": len(s1_sample),
        "total_true_matches": total_true_matches,
        "raw_untruncated_recall": rec_raw / 100.0,
        "recall_cap80": rec_80 / 100.0,
        "recall_cap100": rec_100 / 100.0,
        "recall_cap120": rec_120 / 100.0,
        "mean_candidates_cap80": mean_80,
        "mean_candidates_cap100": mean_100,
        "mean_candidates_cap120": mean_120,
        "zero_candidate_entities": zero_cands_100,
        "signal_hits": dict(signal_hits),
        "runtime_seconds": total_time,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print_log(f"Metrics saved to: {output_dir / 'metrics.json'}")

    log_fp.close()
    print("E004.5 execution completed successfully.")


if __name__ == "__main__":
    main()
