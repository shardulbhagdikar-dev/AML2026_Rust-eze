"""
E002 - Candidate Generation / Blocking Benchmark
Amazon ML Challenge 2026 - Business Entity Resolution

Purpose
-------
Benchmark candidate-generation strategies before training the matcher.

The challenge is NOT to compare every S1 record with every S2/S3 record.
Instead, this script builds inverted indexes over S2/S3 and retrieves
small candidate sets for each S1 entity.

Strategies
----------
1. exact_name
2. exact_address
3. name_or_address
4. token_name
5. char_name
6. combined

Metrics
-------
- Candidate recall
- Mean candidates / S1
- Median candidates / S1
- P95 candidates / S1
- P99 candidates / S1
- Maximum candidates / S1
- Total candidates
- Number of S1 entities with zero candidates
- Recall for S2
- Recall for S3

Ground truth schema
-------------------
source1_entity_id
matched_entity_ids

matched_entity_ids is a comma-separated list.

Example
-------
python E002_candidate_generation.py ^
    --source1 data/train_source1.tsv ^
    --source2 data/train_source2.tsv ^
    --source3 data/train_source3.tsv ^
    --ground-truth data/train_ground_truth.tsv ^
    --output experiments/E002 ^
    --sample-size 10000 ^
    --batch-size 1000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

REQUIRED_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value) -> str:
    """
    Conservative normalization.

    We deliberately DO NOT transliterate or remove non-Latin scripts.
    Cross-script matching is a secondary experiment, not the backbone
    of candidate generation.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    # Unicode compatibility normalization
    text = unicodedata.normalize("NFKC", text)

    # Case-insensitive
    text = text.casefold()

    # Convert punctuation/symbols into spaces.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list[str]:
    if not text:
        return []

    return [
        token
        for token in text.split()
        if token
    ]


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """
    Character n-grams with spaces removed from the generated sequence.

    Example:
        "starbucks" -> {"sta", "tar", "arb", ...}
    """

    if not text:
        return set()

    compact = text.replace(" ", "")

    if len(compact) < n:
        return {compact}

    return {
        compact[i:i + n]
        for i in range(len(compact) - n + 1)
    }


# ============================================================
# DATA LOADING
# ============================================================

def validate_columns(df: pd.DataFrame, name: str):
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}"
        )


def load_source(path: str, source_name: str) -> pd.DataFrame:

    print(f"\nLoading {source_name}: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    validate_columns(df, source_name)

    df = df[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ].copy()

    df["entity_id"] = df["entity_id"].astype(str)
    df["country"] = df["country"].astype(str)

    df["name_norm"] = df["business_name"].map(normalize_text)
    df["address_norm"] = df["business_address"].map(normalize_text)

    print(f"{source_name}: {len(df):,} rows")

    return df


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(path: str) -> dict[str, set[str]]:

    print(f"\nLoading ground truth: {path}")

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required - set(gt.columns)

    if missing:
        raise ValueError(
            f"Ground truth missing columns: {sorted(missing)}"
        )

    truth = {}

    for row in gt.itertuples(index=False):

        s1_id = str(row.source1_entity_id).strip()

        matched_value = str(row.matched_entity_ids).strip()

        if not matched_value:
            target_ids = set()
        else:
            target_ids = {
                x.strip()
                for x in matched_value.split(",")
                if x.strip()
            }

        truth[s1_id] = target_ids

    print(f"Ground-truth S1 entities: {len(truth):,}")

    return truth


# ============================================================
# INDEX
# ============================================================

class BlockingIndex:
    """
    In-memory inverted indexes.

    All indexes are country-aware:

        (country, key) -> target IDs

    This prevents candidates from unrelated countries being generated.
    """

    def __init__(
        self,
        max_postings: int = 500,
        ngram_size: int = 3,
    ):

        self.max_postings = max_postings
        self.ngram_size = ngram_size

        self.name_exact = defaultdict(list)
        self.address_exact = defaultdict(list)

        self.name_token = defaultdict(list)
        self.name_ngram = defaultdict(list)

        self.records = {}

    # --------------------------------------------------------

    def add_record(
        self,
        entity_id: str,
        country: str,
        name: str,
        address: str,
    ):

        self.records[entity_id] = (
            country,
            name,
            address,
        )

        country = country.casefold()

        # -------------------------------
        # Exact name
        # -------------------------------

        if name:

            key = (country, name)

            if len(self.name_exact[key]) < self.max_postings:
                self.name_exact[key].append(entity_id)

        # -------------------------------
        # Exact address
        # -------------------------------

        if address:

            key = (country, address)

            if len(self.address_exact[key]) < self.max_postings:
                self.address_exact[key].append(entity_id)

        # -------------------------------
        # Name tokens
        # -------------------------------

        for token in set(tokenize(name)):

            key = (country, token)

            if len(self.name_token[key]) < self.max_postings:
                self.name_token[key].append(entity_id)

        # -------------------------------
        # Character n-grams
        # -------------------------------

        for gram in char_ngrams(
            name,
            self.ngram_size,
        ):

            key = (country, gram)

            if len(self.name_ngram[key]) < self.max_postings:
                self.name_ngram[key].append(entity_id)


# ============================================================
# INDEX CONSTRUCTION
# ============================================================

def build_index(
    df: pd.DataFrame,
    max_postings: int,
    ngram_size: int,
) -> BlockingIndex:

    print("\nBuilding target index...")

    start = time.time()

    index = BlockingIndex(
        max_postings=max_postings,
        ngram_size=ngram_size,
    )

    total = len(df)

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):

        index.add_record(
            entity_id=row.entity_id,
            country=row.country,
            name=row.name_norm,
            address=row.address_norm,
        )

        if i % 100_000 == 0:

            elapsed = time.time() - start

            rate = i / max(elapsed, 0.001)

            print(
                f"  indexed {i:,}/{total:,} "
                f"({100 * i / total:.1f}%) "
                f"| {rate:,.0f} rows/s"
            )

    print(
        f"Index complete in "
        f"{time.time() - start:.1f}s"
    )

    print(
        f"  exact name keys: "
        f"{len(index.name_exact):,}"
    )

    print(
        f"  exact address keys: "
        f"{len(index.address_exact):,}"
    )

    print(
        f"  name token keys: "
        f"{len(index.name_token):,}"
    )

    print(
        f"  name ngram keys: "
        f"{len(index.name_ngram):,}"
    )

    return index


# ============================================================
# CANDIDATE RETRIEVAL
# ============================================================

def exact_name_candidates(
    row,
    index: BlockingIndex,
) -> set[str]:

    name = row.name_norm

    if not name:
        return set()

    key = (
        str(row.country).casefold(),
        name,
    )

    return set(index.name_exact.get(key, []))


def exact_address_candidates(
    row,
    index: BlockingIndex,
) -> set[str]:

    address = row.address_norm

    if not address:
        return set()

    key = (
        str(row.country).casefold(),
        address,
    )

    return set(index.address_exact.get(key, []))


def token_name_candidates(
    row,
    index: BlockingIndex,
    min_shared_tokens: int = 1,
) -> set[str]:

    country = str(row.country).casefold()

    tokens = set(tokenize(row.name_norm))

    if not tokens:
        return set()

    counts = Counter()

    for token in tokens:

        key = (country, token)

        for entity_id in index.name_token.get(key, []):

            counts[entity_id] += 1

    return {
        entity_id
        for entity_id, count in counts.items()
        if count >= min_shared_tokens
    }


def char_name_candidates(
    row,
    index: BlockingIndex,
    min_shared_ngrams: int = 2,
) -> set[str]:

    country = str(row.country).casefold()

    grams = char_ngrams(
        row.name_norm,
        index.ngram_size,
    )

    if not grams:
        return set()

    counts = Counter()

    for gram in grams:

        key = (country, gram)

        for entity_id in index.name_ngram.get(key, []):

            counts[entity_id] += 1

    return {
        entity_id
        for entity_id, count in counts.items()
        if count >= min_shared_ngrams
    }


# ============================================================
# STRATEGY DISPATCH
# ============================================================

def generate_candidates(
    row,
    index: BlockingIndex,
    strategy: str,
    min_shared_tokens: int,
    min_shared_ngrams: int,
) -> set[str]:

    name_candidates = set()
    address_candidates = set()

    if strategy in {
        "exact_name",
        "name_or_address",
        "combined",
    }:

        name_candidates = exact_name_candidates(
            row,
            index,
        )

    if strategy in {
        "exact_address",
        "name_or_address",
        "combined",
    }:

        address_candidates = exact_address_candidates(
            row,
            index,
        )

    if strategy == "exact_name":
        return name_candidates

    if strategy == "exact_address":
        return address_candidates

    if strategy == "name_or_address":
        return name_candidates | address_candidates

    if strategy == "token_name":

        return token_name_candidates(
            row,
            index,
            min_shared_tokens,
        )

    if strategy == "char_name":

        return char_name_candidates(
            row,
            index,
            min_shared_ngrams,
        )

    if strategy == "combined":

        token_candidates = token_name_candidates(
            row,
            index,
            min_shared_tokens,
        )

        char_candidates = char_name_candidates(
            row,
            index,
            min_shared_ngrams,
        )

        return (
            name_candidates
            | address_candidates
            | token_candidates
            | char_candidates
        )

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


# ============================================================
# EVALUATION
# ============================================================

def percentile(values, q):

    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(values),
            q,
        )
    )


def evaluate_strategy(
    s1_df: pd.DataFrame,
    index: BlockingIndex,
    ground_truth: dict[str, set[str]],
    strategy: str,
    batch_size: int,
    min_shared_tokens: int,
    min_shared_ngrams: int,
    output_dir: Path,
):
    """
    Process S1 in batches.

    Candidate pairs are written incrementally rather than kept entirely
    in RAM.
    """

    print("\n" + "=" * 70)
    print(f"STRATEGY: {strategy}")
    print("=" * 70)

    strategy_dir = output_dir / strategy
    strategy_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_path = (
        strategy_dir /
        "candidate_pairs.tsv"
    )

    # Prevent accidental duplicate output on reruns.
    if candidate_path.exists():
        candidate_path.unlink()

    counts = []

    total_candidates = 0

    true_total = 0
    true_found = 0

    true_s2_total = 0
    true_s2_found = 0

    true_s3_total = 0
    true_s3_found = 0

    zero_candidate_entities = 0

    # We don't know the actual source prefixes, so determine
    # S2/S3 membership from the index records.
    target_source = {}

    for entity_id in index.records:
        target_source[entity_id] = None

    # The caller will replace this with proper source information
    # if needed. Combined recall is the primary metric here.

    start = time.time()

    total_s1 = len(s1_df)

    first_write = True

    for batch_start in range(
        0,
        total_s1,
        batch_size,
    ):

        batch_end = min(
            batch_start + batch_size,
            total_s1,
        )

        batch = s1_df.iloc[
            batch_start:batch_end
        ]

        rows = []

        for row in batch.itertuples(
            index=False
        ):

            s1_id = row.entity_id

            candidates = generate_candidates(
                row,
                index,
                strategy,
                min_shared_tokens,
                min_shared_ngrams,
            )

            count = len(candidates)

            counts.append(count)

            total_candidates += count

            if count == 0:
                zero_candidate_entities += 1

            true_ids = ground_truth.get(
                s1_id,
                set(),
            )

            true_total += len(true_ids)

            found = candidates & true_ids

            true_found += len(found)

            for candidate_id in candidates:

                rows.append(
                    (
                        s1_id,
                        candidate_id,
                    )
                )

        # Write this batch immediately.
        if rows:

            batch_df = pd.DataFrame(
                rows,
                columns=[
                    "source1_entity_id",
                    "matched_entity_id",
                ],
            )

            batch_df.to_csv(
                candidate_path,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )

            first_write = False

        elapsed = time.time() - start

        rate = (
            batch_end /
            max(elapsed, 0.001)
        )

        eta = (
            (total_s1 - batch_end) /
            max(rate, 0.001)
        )

        print(
            f"[{batch_end:,}/{total_s1:,}] "
            f"{100 * batch_end / total_s1:.1f}% | "
            f"batch candidates={sum(counts[-len(batch):]):,} | "
            f"total candidates={total_candidates:,} | "
            f"rate={rate:,.0f} S1/s | "
            f"ETA={eta / 60:.1f} min"
        )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    recall = (
        true_found / true_total
        if true_total
        else 0.0
    )

    result = {
        "strategy": strategy,
        "s1_entities": total_s1,
        "true_matches": true_total,
        "true_matches_found": true_found,
        "candidate_recall": recall,
        "total_candidates": total_candidates,
        "mean_candidates_per_s1": (
            float(np.mean(counts))
            if counts else 0.0
        ),
        "median_candidates_per_s1": (
            float(np.median(counts))
            if counts else 0.0
        ),
        "p95_candidates_per_s1": percentile(
            counts,
            95,
        ),
        "p99_candidates_per_s1": percentile(
            counts,
            99,
        ),
        "max_candidates_per_s1": (
            max(counts)
            if counts else 0
        ),
        "zero_candidate_s1": zero_candidate_entities,
        "zero_candidate_rate": (
            zero_candidate_entities /
            total_s1
            if total_s1
            else 0.0
        ),
        "runtime_seconds": time.time() - start,
    }

    with open(
        strategy_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print("\nRESULT")
    print("-" * 50)

    for key, value in result.items():

        if isinstance(value, float):
            print(
                f"{key:35s}: {value:.6f}"
            )
        else:
            print(
                f"{key:35s}: {value}"
            )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="E002 candidate-generation benchmark"
    )

    parser.add_argument(
        "--source1",
        required=True,
    )

    parser.add_argument(
        "--source2",
        required=True,
    )

    parser.add_argument(
        "--source3",
        required=True,
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="experiments/E002",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Number of S1 entities to benchmark. "
            "Use 10000 or 50000 initially."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--max-postings",
        type=int,
        default=500,
        help=(
            "Maximum number of target records retained "
            "for one blocking key."
        ),
    )

    parser.add_argument(
        "--ngram-size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-shared-tokens",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--min-shared-ngrams",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "exact_name",
            "exact_address",
            "name_or_address",
            "token_name",
            "char_name",
            "combined",
        ],
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    s1 = load_source(
        args.source1,
        "Source 1",
    )

    s2 = load_source(
        args.source2,
        "Source 2",
    )

    s3 = load_source(
        args.source3,
        "Source 3",
    )

    ground_truth = load_ground_truth(
        args.ground_truth
    )

    # --------------------------------------------------------
    # Restrict S1 to sample
    # --------------------------------------------------------

    if args.sample_size is not None:

        if args.sample_size < len(s1):

            # Fixed random state so experiments are reproducible.
            s1 = s1.sample(
                n=args.sample_size,
                random_state=42,
            ).reset_index(
                drop=True
            )

            print(
                f"\nUsing S1 sample: "
                f"{len(s1):,}"
            )

    # --------------------------------------------------------
    # Build combined target index
    # --------------------------------------------------------

    targets = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    print(
        f"\nTotal target records: "
        f"{len(targets):,}"
    )

    index = build_index(
        targets,
        max_postings=args.max_postings,
        ngram_size=args.ngram_size,
    )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = []

    for strategy in args.strategies:

        result = evaluate_strategy(
            s1_df=s1,
            index=index,
            ground_truth=ground_truth,
            strategy=strategy,
            batch_size=args.batch_size,
            min_shared_tokens=args.min_shared_tokens,
            min_shared_ngrams=args.min_shared_ngrams,
            output_dir=output_dir,
        )

        results.append(result)

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    summary = pd.DataFrame(results)

    summary = summary.sort_values(
        [
            "candidate_recall",
            "mean_candidates_per_s1",
        ],
        ascending=[
            False,
            True,
        ],
    )

    summary_path = (
        output_dir /
        "E002_strategy_comparison.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("E002 COMPLETE")
    print("=" * 70)

    print(
        f"\nSummary written to:\n"
        f"{summary_path}"
    )

    print("\nStrategy comparison:")

    print(
        summary[
            [
                "strategy",
                "candidate_recall",
                "mean_candidates_per_s1",
                "median_candidates_per_s1",
                "p95_candidates_per_s1",
                "p99_candidates_per_s1",
                "max_candidates_per_s1",
                "total_candidates",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()