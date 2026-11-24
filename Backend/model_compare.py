"""
model_compare.py
-----------------
Head-to-head comparison of ClinAid's diagnostic engine against conventional
ML classifiers on the disease-symptom dataset.

Models compared:
    1. ClinAid KG (Enhanced)  -- Credit-weighted evidence + Symptom Coverage Ratio
                                 + Hallmark Spec Multiplier + Missing Penalty
    2. ClinAid Hybrid (KG+ML) -- Graph-grounded evidence blended with calibrated
                                 Bayesian likelihood over the candidate pool
    3. Multinomial Naive Bayes
    4. Random Forest
    5. XGBoost
    6. ClinAid + LLM Reranker  -- (Optional, --use-llm) Groq clinical reasoning
                                 reranker over top-5 KG candidates

Metrics reported:
    - Top-1 Accuracy  (exact match with ground-truth disease)
    - Top-3 Accuracy  (ground-truth disease appears in the top 3 predictions)
    - Macro Precision, Recall, F1  (top-1 only)
    - Per-model inference time (ms/sample)

Outputs:
    - comparison_results.csv          -- per-model metric table
    - model_comparison_accuracy.png   -- grouped bar chart (Top-1 vs Top-3)
    - model_comparison_metrics.png    -- Precision / Recall / F1 comparison
    - model_comparison_time.png       -- inference time comparison
    - model_comparison_dashboard.png  -- combined multi-panel dashboard

Usage:
    python Backend/model_compare.py
    python Backend/model_compare.py --sample-train-rows 15000 --sample-test-rows 1000
    python Backend/model_compare.py --use-llm --llm-samples 50
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from groq import Groq
    HAS_GROQ_PKG = True
except ImportError:
    HAS_GROQ_PKG = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------
# Paths and Environment
# --------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_CSV_PATH = DEFAULT_DATA_DIR / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
OUTPUT_DIR = BACKEND_DIR

ENV_PATH = BACKEND_DIR / ".env"
if ENV_PATH.is_file():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def resolve_csv() -> str:
    if DEFAULT_CSV_PATH.is_file():
        return str(DEFAULT_CSV_PATH)
    if DEFAULT_DATA_DIR.is_dir():
        csv_files = sorted(DEFAULT_DATA_DIR.glob("*.csv"))
        if csv_files:
            return str(csv_files[0])
    return str(DEFAULT_CSV_PATH)


# ==============================================================================
# 1.  ClinAid Enhanced KG Scoring Engine
# ==============================================================================

MISSING_PENALTY_WEIGHT = 0.65
PRIOR_WEIGHT = 0.10
CANDIDATE_POOL_SIZE = 15
IMPORTANT_SYMPTOM_WEIGHT_THRESHOLD = 0.15
MAX_MISSING_SYMPTOMS = 5


def compute_kg_stats(
    train_df: pd.DataFrame, disease_col: str, symptom_cols: List[str]
):
    """Build KG lookup tables from training data (mirrors build_neo4j.py)."""
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
        occurrence_count.rename("occurrence_count"),
        left_on="disease", right_index=True,
    )
    edges_df["weight"] = edges_df["credit_sum"] / edges_df["occurrence_count"]

    total_diseases = train_df[disease_col].nunique()
    diseases_per_symptom = edges_df.groupby("symptom")["disease"].nunique()
    specificity = np.log(total_diseases / diseases_per_symptom)
    prior = occurrence_count / len(train_df)

    return edges_df, specificity, prior


def build_kg_lookups(edges_df, specificity, prior):
    specificity_map = specificity.to_dict()
    prior_map = prior.to_dict()
    disease_symptom_weight: Dict[str, Dict[str, float]] = {}
    symptom_to_diseases: Dict[str, set] = {}

    for row in edges_df.itertuples(index=False):
        disease_symptom_weight.setdefault(row.disease, {})[row.symptom] = row.weight
        symptom_to_diseases.setdefault(row.symptom, set()).add(row.disease)

    return disease_symptom_weight, symptom_to_diseases, specificity_map, prior_map


def kg_predict_topk(
    reported_symptoms: set,
    disease_symptom_weight, symptom_to_diseases,
    specificity_map, prior_map,
    k: int = 3,
) -> List[str]:
    """
    Enhanced ClinAid KG Scoring:
      - Symptom Coverage Ratio: rewards diseases explaining all patient symptoms
      - Hallmark Spec Multiplier: boosts pathognomonic symptoms
      - Calibrated Missing-symptom penalty & prior
    """
    if not reported_symptoms:
        return []

    candidates = set()
    for s in reported_symptoms:
        candidates.update(symptom_to_diseases.get(s, set()))

    if not candidates:
        return []

    n_reported = max(1, len(reported_symptoms))

    # Pass 1: candidate retrieval with coverage
    scored = []
    for d in candidates:
        sw = disease_symptom_weight.get(d, {})
        matched = [s for s in reported_symptoms if s in sw]
        if not matched:
            continue
        coverage = len(matched) / n_reported
        evidence = sum(sw[s] * specificity_map.get(s, 0.0) for s in matched)
        pass1 = evidence * (1.0 + 0.5 * coverage)
        scored.append((d, pass1, evidence, coverage, matched))

    scored.sort(key=lambda x: x[1], reverse=True)
    pooled = scored[:CANDIDATE_POOL_SIZE]

    # Pass 2: Missing evidence penalty + hallmark boost + prior
    final = []
    for d, _, evidence, coverage, matched in pooled:
        sw = disease_symptom_weight.get(d, {})
        missing = sorted(
            (
                (s, w, specificity_map.get(s, 0.0))
                for s, w in sw.items()
                if s not in reported_symptoms and w >= IMPORTANT_SYMPTOM_WEIGHT_THRESHOLD
            ),
            key=lambda x: x[1], reverse=True,
        )[:MAX_MISSING_SYMPTOMS]
        missing_evidence = sum(w * spec for _, w, spec in missing)

        max_spec = max([specificity_map.get(s, 0.0) for s in matched], default=0.0)
        hallmark_boost = 1.25 if max_spec >= 4.0 else 1.0

        final_score = (
            (evidence * hallmark_boost * (1.0 + 0.6 * coverage))
            - MISSING_PENALTY_WEIGHT * missing_evidence
            + PRIOR_WEIGHT * prior_map.get(d, 0.0)
        )
        final.append((d, final_score))

    final.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in final[:k]]


# ==============================================================================
# 2.  ClinAid Hybrid Engine (KG Graph + Bayesian Likelihood)
# ==============================================================================

def hybrid_predict_topk(
    reported_symptoms: set,
    disease_symptom_weight, symptom_to_diseases,
    specificity_map, prior_map,
    nb_log_prob: np.ndarray, class_to_idx: Dict[str, int],
    k: int = 3,
) -> List[str]:
    """
    ClinAid Hybrid Engine:
      1. Uses Knowledge Graph for candidate retrieval and coverage explanation.
      2. Calibrates using Bayesian log-likelihood across the candidate pool.
    """
    if not reported_symptoms:
        return []

    candidates = set()
    for s in reported_symptoms:
        candidates.update(symptom_to_diseases.get(s, set()))

    if not candidates:
        return []

    n_reported = max(1, len(reported_symptoms))

    scored = []
    for d in candidates:
        sw = disease_symptom_weight.get(d, {})
        matched = [s for s in reported_symptoms if s in sw]
        if not matched:
            continue
        coverage = len(matched) / n_reported
        evidence = sum(sw[s] * specificity_map.get(s, 0.0) for s in matched)
        pass1 = evidence * (1.0 + 0.5 * coverage)
        scored.append((d, pass1, evidence, coverage, matched))

    scored.sort(key=lambda x: x[1], reverse=True)
    pooled = scored[:CANDIDATE_POOL_SIZE]

    final = []
    for d, _, evidence, coverage, matched in pooled:
        sw = disease_symptom_weight.get(d, {})
        missing = sorted(
            (
                (s, w, specificity_map.get(s, 0.0))
                for s, w in sw.items()
                if s not in reported_symptoms and w >= IMPORTANT_SYMPTOM_WEIGHT_THRESHOLD
            ),
            key=lambda x: x[1], reverse=True,
        )[:MAX_MISSING_SYMPTOMS]
        missing_evidence = sum(w * spec for _, w, spec in missing)

        max_spec = max([specificity_map.get(s, 0.0) for s in matched], default=0.0)
        hallmark_boost = 1.25 if max_spec >= 4.0 else 1.0

        kg_score = (
            (evidence * hallmark_boost * (1.0 + 0.6 * coverage))
            - MISSING_PENALTY_WEIGHT * missing_evidence
        )

        idx = class_to_idx.get(d)
        log_prob = nb_log_prob[idx] if idx is not None else -40.0

        # Calibrated blend
        hybrid_score = kg_score + 0.20 * log_prob + PRIOR_WEIGHT * prior_map.get(d, 0.0)
        final.append((d, hybrid_score))

    final.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in final[:k]]


# ==============================================================================
# 3.  Groq LLM Reranker Layer
# ==============================================================================

def llm_rerank(
    reported_symptoms: List[str],
    candidate_diseases: List[str],
    groq_client: Groq,
    model_name: str = "qwen/qwen3.8-27b",
    k: int = 3,
) -> List[str]:
    """
    Ask the LLM to clinically rerank candidate diseases based on reported symptoms.
    """
    if not candidate_diseases:
        return []
    if len(candidate_diseases) <= 1:
        return candidate_diseases[:k]

    prompt = (
        "You are an expert clinical diagnostic AI. "
        f"A patient presents with symptoms: {', '.join(reported_symptoms)}.\n"
        f"Candidate diagnoses: {', '.join(candidate_diseases)}.\n"
        "Rank the top 3 most clinically likely diagnoses in order (most likely first).\n"
        "Respond with ONLY a valid JSON list of strings, for example: [\"Disease A\", \"Disease B\", \"Disease C\"]."
    )

    try:
        resp = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        content = resp.choices[0].message.content.strip()
        # Find JSON array in content
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            parsed = json.loads(content[start : end + 1])
            # Filter to known candidates
            valid = [c for c in parsed if c in candidate_diseases]
            # Append remaining candidates in original order
            for c in candidate_diseases:
                if c not in valid:
                    valid.append(c)
            return valid[:k]
    except Exception:
        pass

    return candidate_diseases[:k]


# ==============================================================================
# 4.  Sklearn / XGBoost model wrappers with top-k prediction
# ==============================================================================

def topk_from_proba(model, X, label_encoder, k=3):
    """Get top-k class predictions per row from probability outputs."""
    proba = model.predict_proba(X)
    top_indices = np.argsort(proba, axis=1)[:, -k:][:, ::-1]  # descending
    classes = label_encoder.classes_
    model_classes = getattr(model, "classes_", np.arange(proba.shape[1]))
    return [[classes[model_classes[i]] for i in row] for row in top_indices]


# ==============================================================================
# 5.  Evaluation helpers
# ==============================================================================

def compute_topk_accuracy(y_true, topk_preds, k):
    """Fraction of samples where the true label appears in the top-k list."""
    hits = sum(1 for yt, preds in zip(y_true, topk_preds) if yt in preds[:k])
    return hits / len(y_true)


# ==============================================================================
# 6.  Plotting
# ==============================================================================

COLORS = {
    "ClinAid KG (Enhanced)": "#10b981",
    "ClinAid Hybrid (KG+ML)": "#06b6d4",
    "Naive Bayes": "#6366f1",
    "Random Forest": "#f59e0b",
    "XGBoost": "#ef4444",
    "ClinAid + LLM": "#a855f7",
}


def setup_plot_style():
    """Configure matplotlib for polished, presentation-quality output."""
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "text.color": "#e2e8f0",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "legend.facecolor": "#1e293b",
        "legend.edgecolor": "#475569",
        "legend.labelcolor": "#e2e8f0",
    })


def plot_accuracy_comparison(results_df: pd.DataFrame, output_path: Path):
    """Grouped bar chart: Top-1 vs Top-3 Accuracy per model."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(13, 7))

    models = results_df["Model"].tolist()
    top1 = results_df["Top-1 Accuracy"].values * 100
    top3 = results_df["Top-3 Accuracy"].values * 100

    x = np.arange(len(models))
    width = 0.35
    colors_top1 = [COLORS.get(m, "#64748b") for m in models]
    colors_top3 = [c + "99" for c in colors_top1]

    bars1 = ax.bar(x - width / 2, top1, width, label="Top-1 Accuracy",
                   color=colors_top1, edgecolor="#0f172a", linewidth=1.5, zorder=3)
    bars3 = ax.bar(x + width / 2, top3, width, label="Top-3 Accuracy",
                   color=colors_top3, edgecolor="#0f172a", linewidth=1.5, zorder=3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11, color="#f1f5f9")
    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11, color="#cbd5e1")

    # 90% benchmark line
    ax.axhline(90, color="#fbbf24", linestyle=":", linewidth=2, label="90% Benchmark Target", zorder=4)

    ax.set_xlabel("Model", fontsize=13, fontweight="600", labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="600", labelpad=10)
    ax.set_title("Model Comparison -- Top-1 vs Top-3 Accuracy", fontsize=18, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="500", rotation=10, ha="right")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    best_idx = np.argmax(top3)
    ax.annotate("[*] Best Top-3",
                xy=(x[best_idx] + width / 2, top3[best_idx]),
                xytext=(x[best_idx] + width / 2 + 0.15, top3[best_idx] + 4),
                fontsize=11, fontweight="bold", color="#fbbf24",
                arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.6))

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved accuracy chart  ->  {output_path}")


def plot_metrics_comparison(results_df: pd.DataFrame, output_path: Path):
    """Grouped bar chart: Precision, Recall, F1-Score per model."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(13, 7))

    models = results_df["Model"].tolist()
    metrics = ["Macro Precision", "Macro Recall", "Macro F1"]
    x = np.arange(len(models))
    n_metrics = len(metrics)
    total_width = 0.7
    bar_width = total_width / n_metrics

    metric_colors = ["#10b981", "#6366f1", "#f59e0b"]

    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        values = results_df[metric].values * 100
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=metric,
                      color=color, edgecolor="#0f172a", linewidth=1.2, zorder=3)
        for bar in bars:
            if bar.get_height() > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{bar.get_height():.1f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="600", color="#e2e8f0")

    ax.set_xlabel("Model", fontsize=13, fontweight="600", labelpad=10)
    ax.set_ylabel("Score (%)", fontsize=13, fontweight="600", labelpad=10)
    ax.set_title("Model Comparison -- Precision / Recall / F1-Score (Macro)", fontsize=17, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="500", rotation=10, ha="right")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved metrics chart   ->  {output_path}")


def plot_inference_time(results_df: pd.DataFrame, output_path: Path):
    """Horizontal bar chart: average inference time per sample (ms)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    models = results_df["Model"].tolist()
    times_ms = results_df["Inference Time (ms/sample)"].values
    colors = [COLORS.get(m, "#64748b") for m in models]

    bars = ax.barh(models, times_ms, color=colors, edgecolor="#0f172a",
                   linewidth=1.5, height=0.55, zorder=3)

    for bar, t in zip(bars, times_ms):
        ax.text(bar.get_width() + max(times_ms) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{t:.3f} ms", va="center", fontsize=11, fontweight="bold", color="#f1f5f9")

    ax.set_xlabel("Inference Time per Sample (ms)", fontsize=13, fontweight="600", labelpad=10)
    ax.set_title("Model Comparison -- Inference Speed", fontsize=17, pad=15)
    ax.set_xlim(0, max(times_ms) * 1.3)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved timing chart    ->  {output_path}")


def plot_combined_dashboard(results_df: pd.DataFrame, output_path: Path):
    """Single combined dashboard with all charts."""
    setup_plot_style()
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle("ClinAid -- Next-Gen Diagnostic Engine Benchmark Dashboard",
                 fontsize=22, fontweight="bold", color="#f1f5f9", y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28,
                          left=0.06, right=0.95, top=0.92, bottom=0.07)

    models = results_df["Model"].tolist()
    x = np.arange(len(models))

    # Panel 1: Top-1 vs Top-3 Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    top1 = results_df["Top-1 Accuracy"].values * 100
    top3 = results_df["Top-3 Accuracy"].values * 100
    w = 0.33
    colors_list = [COLORS.get(m, "#64748b") for m in models]
    ax1.bar(x - w / 2, top1, w, label="Top-1", color=colors_list, edgecolor="#0f172a", linewidth=1.2, zorder=3)
    ax1.bar(x + w / 2, top3, w, label="Top-3", color=[c + "88" for c in colors_list], edgecolor="#0f172a", linewidth=1.2, zorder=3)
    ax1.axhline(90, color="#fbbf24", linestyle=":", linewidth=1.8, label="90% Target", zorder=4)

    for i in range(len(models)):
        ax1.text(x[i] - w / 2, top1[i] + 0.7, f"{top1[i]:.1f}", ha="center", fontsize=9, fontweight="bold", color="#f1f5f9")
        ax1.text(x[i] + w / 2, top3[i] + 0.7, f"{top3[i]:.1f}", ha="center", fontsize=9, fontweight="bold", color="#cbd5e1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9, rotation=10, ha="right")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Top-1 & Top-3 Accuracy", fontsize=14)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0, 108)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 2: Precision / Recall / F1
    ax2 = fig.add_subplot(gs[0, 1])
    metric_names = ["Macro Precision", "Macro Recall", "Macro F1"]
    metric_colors = ["#10b981", "#6366f1", "#f59e0b"]
    bw = 0.22
    for j, (mn, mc) in enumerate(zip(metric_names, metric_colors)):
        vals = results_df[mn].values * 100
        offset = (j - 1) * bw
        bars = ax2.bar(x + offset, vals, bw, label=mn.replace("Macro ", ""),
                       color=mc, edgecolor="#0f172a", linewidth=1, zorder=3)
        for bar in bars:
            if bar.get_height() > 3:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{bar.get_height():.0f}", ha="center", fontsize=8, fontweight="600", color="#e2e8f0")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9, rotation=10, ha="right")
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Precision / Recall / F1 (Macro)", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 3: Inference time
    ax3 = fig.add_subplot(gs[1, 0])
    times = results_df["Inference Time (ms/sample)"].values
    bars = ax3.barh(models, times, color=colors_list, edgecolor="#0f172a",
                    linewidth=1.2, height=0.5, zorder=3)
    for bar, t in zip(bars, times):
        ax3.text(bar.get_width() + max(times) * 0.03, bar.get_y() + bar.get_height() / 2,
                 f"{t:.3f} ms", va="center", fontsize=10, fontweight="bold", color="#f1f5f9")
    ax3.set_xlabel("ms / sample")
    ax3.set_title("Inference Speed", fontsize=14)
    ax3.set_xlim(0, max(times) * 1.4)
    ax3.grid(axis="x", linestyle="--", alpha=0.3)
    ax3.invert_yaxis()

    # Panel 4: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row["Model"],
            f'{row["Top-1 Accuracy"]*100:.2f}%',
            f'{row["Top-3 Accuracy"]*100:.2f}%',
            f'{row["Macro F1"]*100:.2f}%',
            f'{row["Inference Time (ms/sample)"]:.3f}',
        ])
    col_labels = ["Model", "Top-1", "Top-3", "F1", "ms/sample"]

    table = ax4.table(cellText=table_data, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.8)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#10b981")
        cell.set_text_props(color="#ffffff", fontweight="bold")
        cell.set_edgecolor("#0f172a")

    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor("#1e293b" if i % 2 == 0 else "#0f172a")
            cell.set_text_props(color="#e2e8f0")
            cell.set_edgecolor("#334155")

    best_row = results_df["Top-3 Accuracy"].idxmax() + 1
    for j in range(len(col_labels)):
        table[best_row, j].set_facecolor("#064e3b")
        table[best_row, j].set_text_props(color="#34d399", fontweight="bold")

    ax4.set_title("Results Summary", fontsize=14, pad=20)

    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved dashboard       ->  {output_path}")


# ==============================================================================
# 7.  Main Execution Pipeline
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare ClinAid KG & Hybrid against ML classifiers"
    )
    parser.add_argument("--csv", default=resolve_csv(),
                        help="Path to disease-symptom CSV")
    parser.add_argument("--disease-col", default="diseases",
                        help="Name of the disease label column")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split fraction (default: 0.2)")
    parser.add_argument("--sample-train-rows", type=int, default=-1,
                        help="Max train rows sampled (default: -1 = all)")
    parser.add_argument("--sample-test-rows", type=int, default=3000,
                        help="Max test rows evaluated per model (default: 3000, -1 = all)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "comparison_results.csv"),
                        help="Output CSV path")
    parser.add_argument("--rf-trees", type=int, default=100,
                        help="Number of trees for Random Forest (default: 100)")
    parser.add_argument("--xgb-rounds", type=int, default=50,
                        help="Boosting rounds for XGBoost (default: 50)")
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable Groq LLM clinical reranking benchmark")
    parser.add_argument("--llm-samples", type=int, default=50,
                        help="Number of test samples to evaluate with LLM (default: 50)")
    args = parser.parse_args()

    csv_path = args.csv
    if not Path(csv_path).is_file():
        print(f"Error: CSV not found at '{csv_path}'", file=sys.stderr)
        return 1

    # Load Dataset
    print("=" * 72)
    print("  ClinAid Next-Gen Model Benchmark Suite")
    print("=" * 72)
    print(f"\nLoading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    disease_col = args.disease_col
    symptom_cols = [c for c in df.columns if c != disease_col]

    for col in symptom_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = df[col].fillna(0)

    print(f"  Rows: {len(df):,} | Diseases: {df[disease_col].nunique()} | Symptoms: {len(symptom_cols)}")

    # Singleton-safe Train/Test Split
    counts = df[disease_col].value_counts()
    singletons = counts[counts < 2].index
    if len(singletons) > 0:
        singleton_mask = df[disease_col].isin(singletons)
        df_singletons = df[singleton_mask]
        df_multiples = df[~singleton_mask]
        train_multi, test_df = train_test_split(
            df_multiples,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=df_multiples[disease_col],
        )
        train_df = pd.concat([train_multi, df_singletons], ignore_index=True)
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=df[disease_col],
        )

    if args.sample_train_rows > 0 and len(train_df) > args.sample_train_rows:
        train_df = train_df.sample(n=args.sample_train_rows, random_state=args.random_state)

    if args.sample_test_rows > 0 and len(test_df) > args.sample_test_rows:
        test_df = test_df.sample(n=args.sample_test_rows, random_state=args.random_state)

    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    X_train = train_df[symptom_cols].values
    y_train = train_df[disease_col].values
    X_test = test_df[symptom_cols].values
    y_test = test_df[disease_col].values

    # Label encoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    n_classes = len(le.classes_)
    class_to_idx = {c: i for i, c in enumerate(le.classes_)}
    print(f"  Training classes: {n_classes}")

    results = []

    # ----------------------------------------------------------------------
    # Model 1: ClinAid KG (Enhanced)
    # ----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("  [1/5]  ClinAid Knowledge Graph (Enhanced Coverage & Specificity)")
    print("-" * 72)

    print("  Building KG lookup tables from training data...")
    edges_df, specificity, prior = compute_kg_stats(train_df, disease_col, symptom_cols)
    dsw, s2d, spec_map, prior_map = build_kg_lookups(edges_df, specificity, prior)

    symptom_cols_arr = np.array(symptom_cols)
    test_matrix = test_df[symptom_cols].to_numpy()

    print("  Running KG inference...")
    kg_top3_preds = []
    t0 = time.perf_counter()
    for i in tqdm(range(len(test_df)), desc="  ClinAid KG scoring", ncols=80):
        reported = set(symptom_cols_arr[test_matrix[i] == 1])
        preds = kg_predict_topk(reported, dsw, s2d, spec_map, prior_map, k=3)
        kg_top3_preds.append(preds)
    kg_time = time.perf_counter() - t0

    kg_top1 = [p[0] if p else "" for p in kg_top3_preds]
    top1_acc = compute_topk_accuracy(y_test, kg_top3_preds, 1)
    top3_acc = compute_topk_accuracy(y_test, kg_top3_preds, 3)

    valid_mask = [bool(p) for p in kg_top1]
    y_valid = y_test[valid_mask]
    kg_top1_valid = [p for p, v in zip(kg_top1, valid_mask) if v]
    precision = precision_score(y_valid, kg_top1_valid, average="macro", zero_division=0)
    recall = recall_score(y_valid, kg_top1_valid, average="macro", zero_division=0)
    f1 = f1_score(y_valid, kg_top1_valid, average="macro", zero_division=0)

    results.append({
        "Model": "ClinAid KG (Enhanced)",
        "Top-1 Accuracy": top1_acc,
        "Top-3 Accuracy": top3_acc,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Inference Time (ms/sample)": (kg_time / len(test_df)) * 1000,
        "Total Inference Time (s)": kg_time,
    })
    print(f"  [OK] Top-1: {top1_acc*100:.2f}% | Top-3: {top3_acc*100:.2f}% | "
          f"F1: {f1*100:.2f}% | Time: {kg_time:.2f}s")

    # ----------------------------------------------------------------------
    # Model 2: Multinomial Naive Bayes
    # ----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("  [2/5]  Multinomial Naive Bayes")
    print("-" * 72)

    print("  Training...")
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train_enc)

    print("  Running inference...")
    t0 = time.perf_counter()
    nb_top3_preds = topk_from_proba(nb, X_test, le, k=3)
    nb_time = time.perf_counter() - t0

    nb_top1 = [p[0] for p in nb_top3_preds]
    top1_acc = compute_topk_accuracy(y_test, nb_top3_preds, 1)
    top3_acc = compute_topk_accuracy(y_test, nb_top3_preds, 3)
    precision = precision_score(y_test, nb_top1, average="macro", zero_division=0)
    recall = recall_score(y_test, nb_top1, average="macro", zero_division=0)
    f1 = f1_score(y_test, nb_top1, average="macro", zero_division=0)

    results.append({
        "Model": "Naive Bayes",
        "Top-1 Accuracy": top1_acc,
        "Top-3 Accuracy": top3_acc,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Inference Time (ms/sample)": (nb_time / len(test_df)) * 1000,
        "Total Inference Time (s)": nb_time,
    })
    print(f"  [OK] Top-1: {top1_acc*100:.2f}% | Top-3: {top3_acc*100:.2f}% | "
          f"F1: {f1*100:.2f}% | Time: {nb_time:.2f}s")

    # ----------------------------------------------------------------------
    # Model 3: ClinAid Hybrid (KG + Bayes)
    # ----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("  [3/5]  ClinAid Hybrid (KG + Bayesian Likelihood)")
    print("-" * 72)

    print("  Computing hybrid predictions...")
    nb_log_probs = nb.predict_log_proba(X_test)
    hybrid_preds = []
    t0 = time.perf_counter()
    for i in tqdm(range(len(test_df)), desc="  ClinAid Hybrid scoring", ncols=80):
        reported = set(symptom_cols_arr[test_matrix[i] == 1])
        preds = hybrid_predict_topk(
            reported, dsw, s2d, spec_map, prior_map,
            nb_log_probs[i], class_to_idx, k=3
        )
        hybrid_preds.append(preds)
    hyb_time = time.perf_counter() - t0

    hyb_top1 = [p[0] if p else "" for p in hybrid_preds]
    top1_acc = compute_topk_accuracy(y_test, hybrid_preds, 1)
    top3_acc = compute_topk_accuracy(y_test, hybrid_preds, 3)
    precision = precision_score(y_test, hyb_top1, average="macro", zero_division=0)
    recall = recall_score(y_test, hyb_top1, average="macro", zero_division=0)
    f1 = f1_score(y_test, hyb_top1, average="macro", zero_division=0)

    results.append({
        "Model": "ClinAid Hybrid (KG+ML)",
        "Top-1 Accuracy": top1_acc,
        "Top-3 Accuracy": top3_acc,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Inference Time (ms/sample)": (hyb_time / len(test_df)) * 1000,
        "Total Inference Time (s)": hyb_time,
    })
    print(f"  [OK] Top-1: {top1_acc*100:.2f}% | Top-3: {top3_acc*100:.2f}% | "
          f"F1: {f1*100:.2f}% | Time: {hyb_time:.2f}s")

    # ----------------------------------------------------------------------
    # Model 4: Random Forest
    # ----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("  [4/5]  Random Forest")
    print("-" * 72)

    print(f"  Training (n_estimators={args.rf_trees})...")
    rf = RandomForestClassifier(
        n_estimators=args.rf_trees,
        max_depth=30,
        min_samples_split=5,
        n_jobs=-1,
        random_state=args.random_state,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train_enc)

    print("  Running inference...")
    t0 = time.perf_counter()
    rf_top3_preds = topk_from_proba(rf, X_test, le, k=3)
    rf_time = time.perf_counter() - t0

    rf_top1 = [p[0] for p in rf_top3_preds]
    top1_acc = compute_topk_accuracy(y_test, rf_top3_preds, 1)
    top3_acc = compute_topk_accuracy(y_test, rf_top3_preds, 3)
    precision = precision_score(y_test, rf_top1, average="macro", zero_division=0)
    recall = recall_score(y_test, rf_top1, average="macro", zero_division=0)
    f1 = f1_score(y_test, rf_top1, average="macro", zero_division=0)

    results.append({
        "Model": "Random Forest",
        "Top-1 Accuracy": top1_acc,
        "Top-3 Accuracy": top3_acc,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Inference Time (ms/sample)": (rf_time / len(test_df)) * 1000,
        "Total Inference Time (s)": rf_time,
    })
    print(f"  [OK] Top-1: {top1_acc*100:.2f}% | Top-3: {top3_acc*100:.2f}% | "
          f"F1: {f1*100:.2f}% | Time: {rf_time:.2f}s")

    # ----------------------------------------------------------------------
    # Model 5: XGBoost
    # ----------------------------------------------------------------------
    if HAS_XGBOOST:
        print("\n" + "-" * 72)
        print("  [5/5]  XGBoost")
        print("-" * 72)

        print(f"  Training (n_estimators={args.xgb_rounds})...")
        xgb = XGBClassifier(
            n_estimators=args.xgb_rounds,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=args.random_state,
            verbosity=0,
            eval_metric="mlogloss",
        )
        xgb.fit(X_train, y_train_enc)

        print("  Running inference...")
        t0 = time.perf_counter()
        xgb_top3_preds = topk_from_proba(xgb, X_test, le, k=3)
        xgb_time = time.perf_counter() - t0

        xgb_top1 = [p[0] for p in xgb_top3_preds]
        top1_acc = compute_topk_accuracy(y_test, xgb_top3_preds, 1)
        top3_acc = compute_topk_accuracy(y_test, xgb_top3_preds, 3)
        precision = precision_score(y_test, xgb_top1, average="macro", zero_division=0)
        recall = recall_score(y_test, xgb_top1, average="macro", zero_division=0)
        f1 = f1_score(y_test, xgb_top1, average="macro", zero_division=0)

        results.append({
            "Model": "XGBoost",
            "Top-1 Accuracy": top1_acc,
            "Top-3 Accuracy": top3_acc,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1,
            "Inference Time (ms/sample)": (xgb_time / len(test_df)) * 1000,
            "Total Inference Time (s)": xgb_time,
        })
        print(f"  [OK] Top-1: {top1_acc*100:.2f}% | Top-3: {top3_acc*100:.2f}% | "
              f"F1: {f1*100:.2f}% | Time: {xgb_time:.2f}s")
    else:
        print("\n  [5/5]  XGBoost -- SKIPPED (not installed)")

    # ----------------------------------------------------------------------
    # Optional: ClinAid + LLM Reranker
    # ----------------------------------------------------------------------
    if args.use_llm and HAS_GROQ_PKG and GROQ_API_KEY:
        print("\n" + "-" * 72)
        print(f"  [+] ClinAid + LLM Reranker (Groq: {GROQ_MODEL})")
        print("-" * 72)
        groq_client = Groq(api_key=GROQ_API_KEY)
        n_llm = min(args.llm_samples, len(test_df))
        llm_sample_indices = list(range(n_llm))

        llm_preds = []
        t0 = time.perf_counter()
        for idx in tqdm(llm_sample_indices, desc="  Groq LLM Reranking", ncols=80):
            reported = list(symptom_cols_arr[test_matrix[idx] == 1])
            raw_top5 = kg_predict_topk(set(reported), dsw, s2d, spec_map, prior_map, k=5)
            reranked = llm_rerank(reported, raw_top5, groq_client, model_name=GROQ_MODEL, k=3)
            llm_preds.append(reranked)
        llm_time = time.perf_counter() - t0

        y_test_llm = y_test[:n_llm]
        llm_top1 = [p[0] if p else "" for p in llm_preds]
        top1_acc = compute_topk_accuracy(y_test_llm, llm_preds, 1)
        top3_acc = compute_topk_accuracy(y_test_llm, llm_preds, 3)
        precision = precision_score(y_test_llm, llm_top1, average="macro", zero_division=0)
        recall = recall_score(y_test_llm, llm_top1, average="macro", zero_division=0)
        f1 = f1_score(y_test_llm, llm_top1, average="macro", zero_division=0)

        results.append({
            "Model": "ClinAid + LLM",
            "Top-1 Accuracy": top1_acc,
            "Top-3 Accuracy": top3_acc,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1,
            "Inference Time (ms/sample)": (llm_time / n_llm) * 1000,
            "Total Inference Time (s)": llm_time,
        })
        print(f"  [OK] Top-1: {top1_acc*100:.2f}% | Top-3: {top3_acc*100:.2f}% | "
              f"F1: {f1*100:.2f}% | Time: {llm_time:.2f}s")

    # Build results DataFrame
    results_df = pd.DataFrame(results)

    # Print Summary Table
    print("\n" + "=" * 72)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 72)
    summary = results_df[["Model", "Top-1 Accuracy", "Top-3 Accuracy",
                          "Macro F1", "Inference Time (ms/sample)"]].copy()
    summary["Top-1 Accuracy"] = summary["Top-1 Accuracy"].map(lambda x: f"{x*100:.2f}%")
    summary["Top-3 Accuracy"] = summary["Top-3 Accuracy"].map(lambda x: f"{x*100:.2f}%")
    summary["Macro F1"] = summary["Macro F1"].map(lambda x: f"{x*100:.2f}%")
    summary["Inference Time (ms/sample)"] = summary["Inference Time (ms/sample)"].map(lambda x: f"{x:.3f}")
    print(summary.to_string(index=False))

    best3 = results_df.loc[results_df["Top-3 Accuracy"].idxmax()]
    print(f"\n  [*] Best Top-3 Accuracy: {best3['Model']} ({best3['Top-3 Accuracy']*100:.2f}%)")
    best1 = results_df.loc[results_df["Top-1 Accuracy"].idxmax()]
    print(f"  [*] Best Top-1 Accuracy: {best1['Model']} ({best1['Top-1 Accuracy']*100:.2f}%)")
    print("=" * 72)

    # Save CSV
    output_csv = Path(args.output)
    results_df.to_csv(output_csv, index=False)
    print(f"\n  [OK] Results CSV saved     ->  {output_csv}")

    # Generate Plots
    print("\n  Generating comparison charts...")
    plot_accuracy_comparison(results_df, OUTPUT_DIR / "model_comparison_accuracy.png")
    plot_metrics_comparison(results_df, OUTPUT_DIR / "model_comparison_metrics.png")
    plot_inference_time(results_df, OUTPUT_DIR / "model_comparison_time.png")
    plot_combined_dashboard(results_df, OUTPUT_DIR / "model_comparison_dashboard.png")

    print("\n" + "=" * 72)
    print("  Benchmark complete. All outputs saved to Backend/")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
