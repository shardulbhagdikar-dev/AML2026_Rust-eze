# Amazon ML Challenge 2026: Data Audit & Cardinality Analysis

## 1. Executive Summary

This data audit synthesizes the exploratory data analysis (EDA), ground-truth structure, and true-pair similarity profiling conducted across **12,527,040 total business records** in the training partition.

- **Primary Metric**: Macro-averaged $F_{0.5}$ at the Source-1 entity level.
- **Reference Dataset**: Source-1 ($2,206,821$ entities).
- **Matching Sources**: Source-2 ($5,034,616$ entities) and Source-3 ($5,285,603$ entities).
- **Ground Truth Matches**: $7,638,365$ positive links ($3,693,619$ to S2 and $3,944,746$ to S3).

---

## 2. Dataset Dimensions & Schema

| File | Rows | Columns | Memory / File Size | Format |
| :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 2,206,821 | `entity_id`, `business_name`, `business_address`, `country` | ~210 MB | Tab-separated |
| `train_source2.tsv` | 5,034,616 | `entity_id`, `business_name`, `business_address`, `country` | ~489 MB | Tab-separated |
| `train_source3.tsv` | 5,285,603 | `entity_id`, `business_name`, `business_address`, `country` | ~503 MB | Tab-separated |
| `train_ground_truth.tsv` | 2,206,821 | `source1_entity_id`, `matched_entity_ids` | ~127 MB | Tab-separated |
| `test_source1.tsv` | ~1,750,000 | `entity_id`, `business_name`, `business_address`, `country` | ~175 MB | Tab-separated |
| `test_source2.tsv` | ~5,000,000 | `entity_id`, `business_name`, `business_address`, `country` | ~509 MB | Tab-separated |
| `test_source3.tsv` | ~5,000,000 | `entity_id`, `business_name`, `business_address`, `country` | ~506 MB | Tab-separated |

### Field Quality & Missing Values
- **Missing values**: `0.0000%` across all columns in S1, S2, and S3. Every field contains string data.
- **Duplicate entity IDs**: `0` duplicate IDs within any source file.
- **Countries**:
  - Training: `US` and `India`.
  - Test: `US`, `India`, and an unseen country: `France`.
  - **Constraint**: Normalization and blocking must remain country-agnostic.

---

## 3. Match Cardinality & Singleton Distributions

From the ground truth analysis of all $2,206,821$ Source-1 entities:

| Match Category | S1 Entities | Percentage | Strategic Implication |
| :--- | :--- | :--- | :--- |
| **0 Matches (Singletons)** | **123,247** | **5.58%** | Predicting any match yields $0.0$; empty prediction yields $1.0$. Highly sensitive to false positives. |
| **1 Match** | **119,157** | **5.40%** | Requires balanced high-confidence pair extraction. |
| **2+ Matches** | **1,964,417** | **89.02%** | **Overwhelming majority** of entities have multiple matches across S2 and S3! |
| **Total S1 Entities** | **2,206,821** | **100.0%** | Average matches per non-singleton entity: $\approx 3.66$. |

### Target Match Distribution
- **Total true positive pairs**: $7,638,365$
  - Matches in **Source-2**: $3,693,619$ ($48.36\%$)
  - Matches in **Source-3**: $3,944,746$ ($51.64\%$)
- **Many-to-One / Many-to-Many**:
  - $89\%$ of S1 entities link to multiple records in S2 and S3 simultaneously.
  - S2 and S3 entities can also share identical cluster centers.

---

## 4. True Pair Similarity & Corruption Patterns (E001 & E001.5 Findings)

### 4.1 String Similarity Breakdown (7,638,365 true pairs)
- **Difficult Pairs**: $5,450,464$ pairs ($71.36\%$) exhibit low string similarity under standard fuzzy metrics.
- Causes of divergence:
  1. **Legal Suffix Variations**: e.g., `Pvt Ltd` vs `Private Limited`, `Corp` vs `Corporation`, `LLC`, `Co`.
  2. **Severe Transliteration & Non-Latin Scripts**: Records in Devanagari, Bengali, Telugu, Tamil, Arabic, etc., mapped to Latin equivalents.
  3. **Address Noise**: Landmark-based descriptions ("Near SBI ATM"), missing postal PIN codes, municipality reordering.

### 4.2 Script Independence Invariance (E001.5 Key Insight)
- **Name same script**: $92.78\%$ ($7,086,515$ pairs)
- **Name cross-script**: $6.70\%$ ($511,502$ pairs)
- **Address same script**: $86.53\%$ ($6,609,779$ pairs)
- **Address cross-script**: $4.41\%$ ($337,091$ pairs)
- **Both Name AND Address Cross-Script**: **ONLY 13 PAIRS out of 7.64M ($0.00017\%$)!**

> **Architectural Takeaway**: If the name is in a corrupted or non-Latin script, the address almost always retains original Latin/shared script tokens. Conversely, if the address is non-Latin, the name is in Latin. Multimodal blocking on (Name OR Address) provides near 100% script-coverage ceiling!

---

## 5. Candidate Generation Benchmark (E002 Audit)

Benchmarked on a representative sample of 10,000 S1 entities (with 34,768 true matches):

| Strategy | Candidate Recall | Candidates / S1 | Zero-Candidate Entities | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `exact_name` | $21.42\%$ | 10.1 | $29.89\%$ | Far too low recall on its own. |
| `exact_address` | $8.06\%$ | 0.3 | $76.40\%$ | High precision, tiny coverage. |
| `name_or_address` | $28.22\%$ | 10.4 | $23.52\%$ | Only captures exact overlaps. |
| `token_name` | $47.53\%$ | 1,481.1 | $0.00\%$ | High candidate explosion without adequate recall. |
| `combined` | **$61.83\%$** | **2,279.2** | $0.00\%$ | **Bottleneck**: 38.2% true matches lost, candidate explosion. |

### The Critical Need for E003:
To reach $F_{0.5} \ge 0.90\text{--}0.95+$, candidate recall must be elevated to **$\ge 95\text{--}99\%$** while constraining candidate density to **$\le 50\text{--}80$ candidates/S1**.
This requires:
1. **Rare Token Inverted Index** (filtering common business stop words).
2. **Numeric Token Overlap** (building numbers, PIN codes, phone digits).
3. **Character 3-gram Cosine / TF-IDF Retrieval** via sparse matrix top-$k$.
