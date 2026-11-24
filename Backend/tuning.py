"""
tuning.py
--------------------
Grid-search MISSING_PENALTY_WEIGHT, PRIOR_WEIGHT, and
IMPORTANT_SYMPTOM_WEIGHT_THRESHOLD against a held-out split of your own
dataset, using the SAME scoring formula as diagnosis_api.py:

    final_score(D) = symptom_evidence(D)
                      - MISSING_PENALTY_WEIGHT * missing_evidence(D)
                      + PRIOR_WEIGHT * prior(D)

Method:
    1. Split rows into train/test.
    2. Compute weight, specificity, prior from TRAIN rows only
       (same math as build_neo4j.py) -- this simulates "the KG as it exists".
    3. For each TEST row, treat its symptoms as "what the patient reported"
       and its disease label as ground truth.
    4. Score all candidate diseases and check whether the true disease
       lands in the top-3 (and top-1).
    5. Repeat across a grid of hyperparameter combinations, report the
       combination with the best top-3 accuracy.

Runs entirely in-memory (no Neo4j needed) so the grid search is fast.
Once you pick values here, hardcode them into diagnosis_api.py.

Usage:
    # Run with default dataset from Data/ folder:
    python Backend/tuning.py

    # Or override options:
    python Backend/tuning.py --sample-test-rows 1000 --output Backend/my_results.csv

Requirements:
    pip install pandas numpy scikit-learn tqdm
"""

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_CSV_PATH = DEFAULT_DATA_DIR / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
DEFAULT_OUTPUT_PATH = BACKEND_DIR / "tuning_results.csv"


def resolve_default_csv() -> str:
    """Finds the dataset CSV in the Data directory."""
    if DEFAULT_CSV_PATH.is_file():
        return str(DEFAULT_CSV_PATH)
    if DEFAULT_DATA_DIR.is_dir():
        csv_files = sorted(DEFAULT_DATA_DIR.glob("*.csv"))
        if csv_files:
            return str(csv_files[0])
    return str(DEFAULT_CSV_PATH)


# --------------------------------------------------------------------------
# 1. Build KG stats (weight, specificity, prior) from TRAIN rows only
#    -- mirrors build_neo4j.py's logic exactly, but kept in-memory
# --------------------------------------------------------------------------

def compute_kg_stats(train_df: pd.DataFrame, disease_col: str, symptom_cols: List[str]):
    row_symptom_count = train_df[symptom_cols].sum(axis=1).replace(0, 1)
    credit = train_df[symptom_cols].div(row_symptom_count, axis=0).copy()
    credit[disease_col] = train_df[disease_col].values

    credit_sum_df = credit.groupby(disease_col).sum().copy()
    occurrence_count = train_df[disease_col].value_counts()

    edges_df = (
        credit_sum_df.reset_index()
        .melt(id_vars=disease_col, var_name="symptom", value_name="credit_sum")
    )
    edges_df = edges_df[edges_df["credit_sum"] > 0].reset_index(drop=True)
    edges_df = edges_df.rename(columns={disease_col: "disease"})
    edges_df = edges_df.merge(
        occurrence_count.rename("occurrence_count"), left_on="disease", right_index=True
    )
    edges_df["weight"] = edges_df["credit_sum"] / edges_df["occurrence_count"]

    total_diseases = train_df[disease_col].nunique()
    diseases_per_symptom = edges_df.groupby("symptom")["disease"].nunique()
    specificity = np.log(total_diseases / diseases_per_symptom)

    prior = occurrence_count / len(train_df)

    return edges_df, specificity, prior


# --------------------------------------------------------------------------
# 2. Build fast in-memory lookup structures
# --------------------------------------------------------------------------

def build_lookup_structures(edges_df: pd.DataFrame, specificity: pd.Series, prior: pd.Series):
    specificity_map: Dict[str, float] = specificity.to_dict()
    prior_map: Dict[str, float] = prior.to_dict()

    # disease -> {symptom: weight}
    disease_symptom_weight: Dict[str, Dict[str, float]] = {}
    # symptom -> set of diseases (reverse index, to find candidates fast)
    symptom_to_diseases: Dict[str, set] = {}

    for row in edges_df.itertuples(index=False):
        disease_symptom_weight.setdefault(row.disease, {})[row.symptom] = row.weight
        symptom_to_diseases.setdefault(row.symptom, set()).add(row.disease)

    return disease_symptom_weight, symptom_to_diseases, specificity_map, prior_map


# --------------------------------------------------------------------------
# 3. Precompute, per test row, the candidate pool + symptom_evidence
#    (this part is independent of the hyperparameters, so we do it ONCE)
# --------------------------------------------------------------------------

def precompute_row_candidates(
    test_df: pd.DataFrame,
    disease_col: str,
    symptom_cols: List[str],
    disease_symptom_weight: Dict[str, Dict[str, float]],
    symptom_to_diseases: Dict[str, set],
    specificity_map: Dict[str, float],
    prior_map: Dict[str, float],
    candidate_pool_size: int,
):
    """
    Returns a list of dicts, one per test row:
        {
          "true_disease": str,
          "candidates": [
              {
                "disease": str,
                "symptom_evidence": float,
                "prior": float,
                "missing_pool": [(symptom, weight, specificity), ...] sorted desc by weight,
              }, ...
          ]
        }
    """
    precomputed = []

    symptom_cols_arr = np.array(symptom_cols)
    test_matrix = test_df[symptom_cols].to_numpy()
    test_diseases = test_df[disease_col].to_numpy()

    for i in tqdm(range(len(test_df)), desc="Precomputing candidates"):
        true_disease = test_diseases[i]
        row_vals = test_matrix[i]
        reported = set(symptom_cols_arr[row_vals == 1])

        if not reported:
            continue

        # Union of diseases connected to any reported symptom
        candidate_diseases = set()
        for s in reported:
            candidate_diseases.update(symptom_to_diseases.get(s, set()))

        if not candidate_diseases:
            continue

        scored = []
        for d in candidate_diseases:
            sym_weights = disease_symptom_weight[d]
            evidence = sum(
                sym_weights[s] * specificity_map.get(s, 0.0)
                for s in reported if s in sym_weights
            )
            scored.append((d, evidence))

        # Mirror production's pass-1 cutoff: keep top-N by raw symptom_evidence
        scored.sort(key=lambda x: x[1], reverse=True)
        pooled = scored[:candidate_pool_size]

        candidates = []
        for d, evidence in pooled:
            sym_weights = disease_symptom_weight[d]
            missing_pool = sorted(
                (
                    (s, w, specificity_map.get(s, 0.0))
                    for s, w in sym_weights.items() if s not in reported
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            candidates.append({
                "disease": d,
                "symptom_evidence": evidence,
                "prior": prior_map.get(d, 0.0),
                "missing_pool": missing_pool,
            })

        precomputed.append({"true_disease": true_disease, "candidates": candidates})

    return precomputed


# --------------------------------------------------------------------------
# 4. Grid search
# --------------------------------------------------------------------------

def evaluate_combo(
    precomputed: List[dict],
    threshold: float,
    penalty_weight: float,
    prior_weight: float,
    max_missing: int,
) -> Tuple[float, float]:
    """Returns (top1_accuracy, top3_accuracy) for one hyperparameter combo."""
    top1_hits = 0
    top3_hits = 0
    n = len(precomputed)

    for item in precomputed:
        true_disease = item["true_disease"]
        scored = []
        for c in item["candidates"]:
            missing = [m for m in c["missing_pool"] if m[1] >= threshold][:max_missing]
            missing_evidence = sum(w * spec for _, w, spec in missing)
            final_score = (
                c["symptom_evidence"]
                - penalty_weight * missing_evidence
                + prior_weight * c["prior"]
            )
            scored.append((c["disease"], final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top3 = [d for d, _ in scored[:3]]

        if top3 and top3[0] == true_disease:
            top1_hits += 1
        if true_disease in top3:
            top3_hits += 1

    return top1_hits / n, top3_hits / n


def grid_search(
    precomputed: List[dict],
    thresholds: List[float],
    penalty_weights: List[float],
    prior_weights: List[float],
    max_missing: int,
):
    results = []
    combos = list(itertools.product(thresholds, penalty_weights, prior_weights))

    for threshold, penalty_weight, prior_weight in tqdm(combos, desc="Grid search"):
        top1_acc, top3_acc = evaluate_combo(
            precomputed, threshold, penalty_weight, prior_weight, max_missing
        )
        results.append({
            "threshold": threshold,
            "penalty_weight": penalty_weight,
            "prior_weight": prior_weight,
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
        })

    return pd.DataFrame(results).sort_values(
        ["top3_accuracy", "top1_accuracy"], ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------

def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",")]


def main():
    default_csv = resolve_default_csv()
    default_output = str(DEFAULT_OUTPUT_PATH)

    parser = argparse.ArgumentParser(description="Tune KG scoring hyperparameters")
    parser.add_argument(
        "--csv",
        default=default_csv,
        help=f"Path to the disease-symptom CSV file (default: {default_csv})",
    )
    parser.add_argument(
        "--disease-col",
        default="diseases",
        help="Name of the disease column (default: 'diseases')",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test split (default: 0.2)",
    )
    parser.add_argument(
        "--sample-test-rows",
        type=int,
        default=3000,
        help="Cap test rows evaluated per grid combo, for speed. Use -1 for all. (default: 3000)",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=15,
        help="Number of disease candidates kept in first pass (default: 15)",
    )
    parser.add_argument(
        "--max-missing-symptoms",
        type=int,
        default=5,
        help="Max missing symptoms evaluated per candidate (default: 5)",
    )
    parser.add_argument(
        "--threshold-grid",
        default="0.05,0.10,0.15,0.20,0.30",
        help="Comma-separated thresholds to evaluate",
    )
    parser.add_argument(
        "--penalty-grid",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated missing penalty weights",
    )
    parser.add_argument(
        "--prior-grid",
        default="0.0,0.05,0.1,0.2",
        help="Comma-separated prior weights",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Path to output CSV results file (default: {default_output})",
    )
    args = parser.parse_args()

    if not args.csv or not Path(args.csv).is_file():
        print(f"Error: CSV dataset not found at '{args.csv}'.", file=sys.stderr)
        print("Please ensure the CSV file is located in the Data/ folder or specify --csv.", file=sys.stderr)
        return 1

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    disease_col = args.disease_col

    if disease_col not in df.columns:
        print(f"Error: Disease column '{disease_col}' not found in CSV columns.", file=sys.stderr)
        return 1

    symptom_cols = [c for c in df.columns if c != disease_col]

    if not all(np.issubdtype(df[col].dtype, np.number) for col in symptom_cols):
        df[symptom_cols] = df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    else:
        df[symptom_cols] = df[symptom_cols].fillna(0)

    print(f"Total rows: {len(df):,} | diseases: {df[disease_col].nunique():,} | symptoms: {len(symptom_cols):,}")

    # Stratified split where possible; fall back to plain split if some
    # diseases have too few rows to stratify (sklearn requires >=2 per class).
    counts = df[disease_col].value_counts()
    can_stratify = (counts >= 2).all()
    stratify_col = df[disease_col] if can_stratify else None
    if not can_stratify:
        print("Warning: some diseases have <2 rows -- disabling stratified split.")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=stratify_col
    )
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    if args.sample_test_rows > 0 and len(test_df) > args.sample_test_rows:
        test_df = test_df.sample(n=args.sample_test_rows, random_state=args.random_state)
        print(f"Sampling down to {len(test_df):,} test rows for grid-search speed.")

    print("\nComputing KG stats (weight, specificity, prior) from TRAIN split...")
    edges_df, specificity, prior = compute_kg_stats(train_df, disease_col, symptom_cols)

    print("Building in-memory lookup structures...")
    disease_symptom_weight, symptom_to_diseases, specificity_map, prior_map = build_lookup_structures(
        edges_df, specificity, prior
    )

    print("\nPrecomputing per-row candidate pools (independent of hyperparameters)...")
    precomputed = precompute_row_candidates(
        test_df, disease_col, symptom_cols,
        disease_symptom_weight, symptom_to_diseases, specificity_map, prior_map,
        candidate_pool_size=args.candidate_pool_size,
    )
    print(f"Usable test rows (had >=1 known symptom with >=1 candidate disease): {len(precomputed):,}")

    thresholds = parse_float_list(args.threshold_grid)
    penalty_weights = parse_float_list(args.penalty_grid)
    prior_weights = parse_float_list(args.prior_grid)
    total_combos = len(thresholds) * len(penalty_weights) * len(prior_weights)
    print(f"\nGrid search over {total_combos} combinations "
          f"({len(thresholds)} thresholds x {len(penalty_weights)} penalties x {len(prior_weights)} priors)...")

    results_df = grid_search(
        precomputed, thresholds, penalty_weights, prior_weights, args.max_missing_symptoms
    )

    # Baseline: symptom_evidence only (penalty=0, prior=0), ignoring threshold entirely
    baseline_top1, baseline_top3 = evaluate_combo(precomputed, thresholds[0], 0.0, 0.0, args.max_missing_symptoms)

    print("\n" + "=" * 70)
    print(f"Baseline (symptom_evidence only, no penalty/prior): "
          f"top1={baseline_top1:.4f}  top3={baseline_top3:.4f}")
    print("=" * 70)
    print("\nTop 10 hyperparameter combinations by top-3 accuracy:\n")
    print(results_df.head(10).to_string(index=False))

    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("BEST COMBINATION:")
    print(f"  IMPORTANT_SYMPTOM_WEIGHT_THRESHOLD = {best['threshold']}")
    print(f"  MISSING_PENALTY_WEIGHT             = {best['penalty_weight']}")
    print(f"  PRIOR_WEIGHT                        = {best['prior_weight']}")
    print(f"  -> top1_accuracy = {best['top1_accuracy']:.4f}, top3_accuracy = {best['top3_accuracy']:.4f}")
    print("=" * 70)

    results_df.to_csv(args.output, index=False)
    print(f"\nFull grid results saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())