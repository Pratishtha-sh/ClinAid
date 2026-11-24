"""
build_neo4j.py
--------------
Builds a Disease -> Symptom knowledge graph in Neo4j from a binary
disease-symptom matrix (Mendeley-style: one row per case, columns =
symptoms, values = 0/1, plus a 'diseases' column).

Graph model:
    (:Disease {name, occurrence_count, prior})
        -[:HAS_SYMPTOM {credit_sum, weight}]->
    (:Symptom {name, specificity})

Credentials:
    Loaded automatically from Backend/.env (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD).

Dataset:
    Defaults to the CSV file located in the project's Data/ directory
    (e.g., Data/Final_Augmented_dataset_Diseases_and_Symptoms.csv).

Usage:
    # Run using defaults from Backend/.env and Data/ folder:
    python Backend/build_neo4j.py

    # Or with explicit overrides:
    python Backend/build_neo4j.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# --------------------------------------------------------------------------
# Paths & Environment resolution
# --------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_ENV_PATH = BACKEND_DIR / ".env"
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_CSV_PATH = DEFAULT_DATA_DIR / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"


def load_env(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Loads environment variables from .env file if it exists."""
    if not env_path.is_file():
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val


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
# 1. Data preparation (pandas)
# --------------------------------------------------------------------------

def load_and_prepare(
    csv_path: str, disease_col: str = "diseases"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads the raw binary matrix and computes:
      - edges_df: one row per (disease, symptom) with credit_sum & weight
      - priors_df: one row per disease with occurrence_count and prior
      - symptoms_df: one row per symptom with IDF-style specificity

    Returns (edges_df, priors_df, symptoms_df)
    """
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if disease_col not in df.columns:
        raise ValueError(
            f"Expected a '{disease_col}' column, found columns: {list(df.columns)[:5]}..."
        )

    symptom_cols: List[str] = [c for c in df.columns if c != disease_col]
    print(f"Rows: {len(df):,} | Disease column: '{disease_col}' | Symptom columns: {len(symptom_cols):,}")

    # Ensure symptom columns are numeric 0/1 (coerce if needed, fill NaN with 0)
    if not all(np.issubdtype(df[col].dtype, np.number) for col in symptom_cols):
        df[symptom_cols] = df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    else:
        df[symptom_cols] = df[symptom_cols].fillna(0)

    # Number of symptoms present in each row
    row_symptom_count = df[symptom_cols].sum(axis=1)
    # Avoid division by zero for any all-zero rows
    safe_row_count = row_symptom_count.replace(0, 1)

    print("Computing per-row credit (1 / num_symptoms_in_row)...")
    credit = df[symptom_cols].div(safe_row_count, axis=0).copy()
    credit[disease_col] = df[disease_col].values

    print("Aggregating credit_sum per (disease, symptom)...")
    credit_sum_df = credit.groupby(disease_col).sum().copy()

    print("Computing occurrence_count per disease...")
    occurrence_count = df[disease_col].value_counts()

    print("Melting into long edge list (disease, symptom, credit_sum)...")
    edges_df = (
        credit_sum_df
        .reset_index()
        .melt(id_vars=disease_col, var_name="symptom", value_name="credit_sum")
    )
    # Drop zero-credit pairs (i.e. symptom never observed for that disease)
    edges_df = edges_df[edges_df["credit_sum"] > 0].reset_index(drop=True)
    edges_df = edges_df.rename(columns={disease_col: "disease"})

    priors_df = occurrence_count.rename_axis("disease").reset_index(name="occurrence_count")
    total_rows = len(df)
    priors_df["prior"] = priors_df["occurrence_count"] / total_rows

    # Attach occurrence_count to edges so we can precompute weight directly
    edges_df = edges_df.merge(priors_df[["disease", "occurrence_count"]], on="disease", how="left")
    edges_df["weight"] = edges_df["credit_sum"] / edges_df["occurrence_count"]

    print("Computing symptom specificity (IDF-style: log(total_diseases / diseases_containing_symptom))...")
    total_diseases = priors_df["disease"].nunique()
    diseases_per_symptom = edges_df.groupby("symptom")["disease"].nunique()
    specificity = np.log(total_diseases / diseases_per_symptom)
    symptoms_df = specificity.rename("specificity").reset_index()

    print(f"Prepared {len(edges_df):,} disease-symptom edges across {len(priors_df):,} diseases "
          f"and {len(symptoms_df):,} symptoms.")
    return edges_df, priors_df, symptoms_df


# --------------------------------------------------------------------------
# 2. Neo4j loading
# --------------------------------------------------------------------------

class KGBuilder:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Uniqueness constraints -> also creates backing indexes for fast MERGE."""
        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT disease_name IF NOT EXISTS "
                "FOR (d:Disease) REQUIRE d.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT symptom_name IF NOT EXISTS "
                "FOR (s:Symptom) REQUIRE s.name IS UNIQUE"
            )
        print("Constraints ensured (Disease.name, Symptom.name uniqueness).")

    def load_disease_priors(self, priors_df: pd.DataFrame, batch_size: int = 500):
        records = priors_df.to_dict("records")
        query = """
        UNWIND $rows AS row
        MERGE (d:Disease {name: row.disease})
        SET d.occurrence_count = row.occurrence_count,
            d.prior = row.prior
        """
        self._run_batched(query, records, batch_size, desc="Loading Disease nodes")

    def load_symptom_specificity(self, symptoms_df: pd.DataFrame, batch_size: int = 500):
        records = symptoms_df.to_dict("records")
        query = """
        UNWIND $rows AS row
        MERGE (s:Symptom {name: row.symptom})
        SET s.specificity = row.specificity
        """
        self._run_batched(query, records, batch_size, desc="Loading Symptom specificity")

    def load_edges(self, edges_df: pd.DataFrame, batch_size: int = 2000):
        records = edges_df.to_dict("records")
        query = """
        UNWIND $rows AS row
        MERGE (d:Disease {name: row.disease})
        MERGE (s:Symptom {name: row.symptom})
        MERGE (d)-[r:HAS_SYMPTOM]->(s)
        SET r.credit_sum = row.credit_sum,
            r.weight = row.weight
        """
        self._run_batched(query, records, batch_size, desc="Loading HAS_SYMPTOM edges")

    def _run_batched(self, query: str, records: list, batch_size: int, desc: str):
        with self.driver.session(database=self.database) as session:
            for i in tqdm(range(0, len(records), batch_size), desc=desc):
                batch = records[i:i + batch_size]
                session.run(query, rows=batch)

    def sanity_check(self):
        with self.driver.session(database=self.database) as session:
            n_diseases = session.run("MATCH (d:Disease) RETURN count(d) AS c").single()["c"]
            n_symptoms = session.run("MATCH (s:Symptom) RETURN count(s) AS c").single()["c"]
            n_edges = session.run("MATCH ()-[r:HAS_SYMPTOM]->() RETURN count(r) AS c").single()["c"]
            print(f"\nGraph summary:")
            print(f"  Disease nodes : {n_diseases:,}")
            print(f"  Symptom nodes : {n_symptoms:,}")
            print(f"  HAS_SYMPTOM edges : {n_edges:,}")


# --------------------------------------------------------------------------
# 3. Main
# --------------------------------------------------------------------------

def main():
    # Load .env from Backend folder by default
    load_env(DEFAULT_ENV_PATH)

    default_csv = resolve_default_csv()

    parser = argparse.ArgumentParser(description="Build Disease-Symptom KG in Neo4j")
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
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI (defaults to NEO4J_URI from Backend/.env)",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username (defaults to NEO4J_USERNAME from Backend/.env)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j password (defaults to NEO4J_PASSWORD from Backend/.env)",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("NEO4J_DATABASE", "c07bfbbe"),
        help="Neo4j database name (default: 'c07bfbbe')",
    )
    parser.add_argument("--node-batch-size", type=int, default=500)
    parser.add_argument("--edge-batch-size", type=int, default=2000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run pandas prep and print stats, skip Neo4j writes",
    )
    args = parser.parse_args()

    # Validation
    if not args.csv or not Path(args.csv).is_file():
        print(f"Error: CSV dataset not found at '{args.csv}'.", file=sys.stderr)
        print("Please ensure the CSV file is in the Data/ folder or specify --csv.", file=sys.stderr)
        return 1

    if not args.dry_run and not args.password:
        print(
            "Error: Neo4j password not provided.\n"
            "Please ensure NEO4J_PASSWORD is set in Backend/.env or pass --password.",
            file=sys.stderr,
        )
        return 1

    print(f"Dataset path: {args.csv}")
    edges_df, priors_df, symptoms_df = load_and_prepare(args.csv, disease_col=args.disease_col)

    print("\nSample edges:")
    print(edges_df.sort_values("weight", ascending=False).head(10).to_string(index=False))

    print("\nSample disease priors:")
    print(priors_df.sort_values("occurrence_count", ascending=False).head(10).to_string(index=False))

    print("\nSample symptom specificity (higher = more discriminative):")
    print(symptoms_df.sort_values("specificity", ascending=False).head(10).to_string(index=False))

    if args.dry_run:
        print("\n--dry-run set: skipping Neo4j load.")
        return 0

    print(f"\nConnecting to Neo4j at {args.uri} (user: {args.user})...")
    builder = KGBuilder(args.uri, args.user, args.password, args.database)
    try:
        builder.create_constraints()
        builder.load_disease_priors(priors_df, batch_size=args.node_batch_size)
        builder.load_symptom_specificity(symptoms_df, batch_size=args.node_batch_size)
        builder.load_edges(edges_df, batch_size=args.edge_batch_size)
        builder.sanity_check()
        print("\nKnowledge graph build completed successfully.")
        return 0
    finally:
        builder.close()


if __name__ == "__main__":
    sys.exit(main())