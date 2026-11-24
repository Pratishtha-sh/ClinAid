"""
diagnosis_api.py
-----------------
ClinAid backend API.

Flow:
    Frontend --(symptoms + optional patient context)--> /diagnose
        -> Query Neo4j KG for top-3 candidate diseases with a full,
           inspectable evidence trail (this is the "explainability" layer)
        -> Pass that evidence trail + patient context to the LLM (Groq)
        -> LLM reasons ONLY over the given evidence (grounding / anti-hallucination)
        -> Return both the raw KG evidence and the LLM's structured explanation

Run:
    uvicorn diagnosis_api:app --reload --port 8000

Env vars required (.env):
    NEO4J_URI=neo4j+s://...
    NEO4J_USERNAME=neo4j (or NEO4J_USER)
    NEO4J_PASSWORD=...
    NEO4J_DATABASE=neo4j                     # optional, default: neo4j
    GROQ_API_KEY=gsk_...
    GROQ_MODEL=llama-3.3-70b-versatile       # optional, default: llama-3.3-70b-versatile
    ALLOWED_ORIGINS=http://localhost:3000    # optional, comma-separated, default: *
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from dotenv import load_dotenv
from groq import Groq

from dynamic_kg import resolve_unmatched_symptoms, approve_and_add_to_kg

# --------------------------------------------------------------------------
# Load Environment Variables from Backend/.env
# --------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
ENV_PATH = BACKEND_DIR / ".env"
if ENV_PATH.is_file():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinaid")

# --------------------------------------------------------------------------
# Config & Connections
# --------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# Support both NEO4J_USERNAME (used in .env) and NEO4J_USER
NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "c07bfbbe")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
FALLBACK_GROQ_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
ALLOW_CREDENTIALS = False if ALLOWED_ORIGINS == ["*"] else True

# Scoring hyperparameters — tuned on dataset validation split (top-3 accuracy: 91.30%, top-1: 69.47%)
MISSING_PENALTY_WEIGHT = 0.75     # how harshly we punish absent "expected" symptoms
PRIOR_WEIGHT = 0.1                # nudge from base-rate prevalence
CANDIDATE_POOL_SIZE = 15          # widen net before applying missing-symptom penalty
IMPORTANT_SYMPTOM_WEIGHT_THRESHOLD = 0.15   # only "expected" symptoms above this weight count toward missing_evidence
MAX_MISSING_SYMPTOMS_RETURNED = 5

# --------------------------------------------------------------------------
# Neo4j access layer
# --------------------------------------------------------------------------

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_candidate_pool(symptoms: List[str]) -> Dict[str, Any]:
    """
    First pass: find diseases connected to any of the input symptoms
    (case-insensitive match) and compute raw symptom_evidence for each.
    Also reports which input symptoms had no match in the KG at all.
    """
    query = """
    UNWIND $symptoms AS inputSymptom
    MATCH (s:Symptom)
    WHERE toLower(s.name) = toLower(inputSymptom)
    WITH inputSymptom, s
    OPTIONAL MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s)
    RETURN inputSymptom, s.name AS matchedSymptom, s.specificity AS specificity,
           d.name AS disease, d.prior AS prior, r.weight AS weight
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        rows = list(session.run(query, symptoms=symptoms))

    matched_input_symptoms = set()
    per_disease: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if row["matchedSymptom"] is not None:
            matched_input_symptoms.add(row["inputSymptom"].lower())
        if row["disease"] is None:
            continue  # symptom matched in KG but no disease edge (shouldn't normally happen)

        d = row["disease"]
        entry = per_disease.setdefault(d, {
            "disease": d,
            "prior": row["prior"] or 0.0,
            "matched_symptoms": [],
            "symptom_evidence": 0.0,
        })
        weight = row["weight"] or 0.0
        specificity = row["specificity"] or 0.0
        contribution = weight * specificity
        entry["matched_symptoms"].append({
            "symptom": row["matchedSymptom"],
            "weight": round(weight, 4),
            "specificity": round(specificity, 4),
            "contribution": round(contribution, 4),
        })
        entry["symptom_evidence"] += contribution

    unmatched_symptoms = [s for s in symptoms if s.lower() not in matched_input_symptoms]

    ranked = sorted(per_disease.values(), key=lambda x: x["symptom_evidence"], reverse=True)
    return {
        "candidates": ranked[:CANDIDATE_POOL_SIZE],
        "unmatched_symptoms": unmatched_symptoms,
    }


def get_missing_evidence(disease: str, reported_symptoms: List[str]) -> Dict[str, Any]:
    """
    Second pass (only for the pooled candidates): find that disease's
    other well-associated symptoms the patient did NOT report, so we can
    penalize diseases missing their "expected" symptoms.
    """
    query = """
    MATCH (d:Disease {name: $disease})-[r:HAS_SYMPTOM]->(s:Symptom)
    WHERE r.weight >= $threshold AND NOT toLower(s.name) IN $reported
    RETURN s.name AS symptom, r.weight AS weight, s.specificity AS specificity
    ORDER BY r.weight DESC
    LIMIT $limit
    """
    reported_lower = [s.lower() for s in reported_symptoms]
    with driver.session(database=NEO4J_DATABASE) as session:
        rows = list(session.run(
            query,
            disease=disease,
            threshold=IMPORTANT_SYMPTOM_WEIGHT_THRESHOLD,
            reported=reported_lower,
            limit=MAX_MISSING_SYMPTOMS_RETURNED,
        ))

    missing = [{
        "symptom": r["symptom"],
        "weight": round(r["weight"] or 0.0, 4),
        "specificity": round(r["specificity"] or 0.0, 4),
    } for r in rows]

    missing_evidence = sum(m["weight"] * m["specificity"] for m in missing)
    return {"missing_symptoms": missing, "missing_evidence": missing_evidence}


def score_diseases(symptoms: List[str]) -> Dict[str, Any]:
    pool = get_candidate_pool(symptoms)
    candidates = pool["candidates"]
    n_reported = max(1, len(symptoms))

    for c in candidates:
        miss = get_missing_evidence(c["disease"], symptoms)
        c["missing_symptoms"] = miss["missing_symptoms"]
        c["missing_evidence"] = round(miss["missing_evidence"], 4)

        # Coverage ratio: fraction of patient presentation explained by this disease
        n_matched = len(c.get("matched_symptoms", []))
        coverage_ratio = n_matched / n_reported

        # Hallmark boost for high-specificity symptoms
        max_spec = max([m.get("specificity", 0.0) for m in c.get("matched_symptoms", [])], default=0.0)
        hallmark_boost = 1.25 if max_spec >= 4.0 else 1.0

        c["final_score"] = round(
            (c["symptom_evidence"] * hallmark_boost * (1.0 + 0.6 * coverage_ratio))
            - MISSING_PENALTY_WEIGHT * miss["missing_evidence"]
            + PRIOR_WEIGHT * c["prior"],
            4,
        )
        c["symptom_evidence"] = round(c["symptom_evidence"], 4)

    top3 = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:3]
    return {
        "top_diseases": top3,
        "unmatched_symptoms": pool["unmatched_symptoms"],
        "input_symptoms": symptoms,
    }


# --------------------------------------------------------------------------
# LLM layer (Groq) — reasons ONLY over the KG evidence + patient-provided context
# --------------------------------------------------------------------------

llm_client: Optional[Groq] = None
if GROQ_API_KEY:
    llm_client = Groq(api_key=GROQ_API_KEY)
else:
    logger.warning("GROQ_API_KEY not found in environment or Backend/.env")

SYSTEM_PROMPT = """You are a clinical-reasoning assistant inside ClinAid, a symptom-checker tool.

You are given:
1. A ranked list of up to 3 candidate diseases, each with a transparent evidence
   trail computed from a medical knowledge graph: matched symptoms (with their
   association weight and specificity), missing expected symptoms, and a
   disease prior.
2. Free-text context the patient provided (e.g. duration, severity, history).

Your job:
- Reason ONLY from the evidence given to you. Do not introduce new symptoms,
   diseases, or statistics that are not present in the provided evidence or
   general well-established medical knowledge needed to interpret it.
- For each candidate disease, explain in plain language WHY it is or isn't a
   good fit, explicitly referencing the matched/missing symptoms and how the
   patient's free-text context supports or weakens that candidate.
- Rank the 3 candidates in your own final assessment (it may differ from the
   raw KG order if the patient's context changes the picture) and justify any
   reordering explicitly.
- Always include a clear, prominent disclaimer that this is not a medical
   diagnosis and the patient should consult a licensed clinician, especially
   for urgent or worsening symptoms.
- Respond ONLY with valid JSON, matching this schema:

{
  "assessment": [
    {
      "disease": "string",
      "rank": 1,
      "confidence_level": "low | moderate | high",
      "reasoning": "string explaining fit using the given evidence and patient context",
      "supporting_evidence": ["short evidence bullet strings, drawn from matched_symptoms"],
      "concerns_or_gaps": ["short bullet strings, drawn from missing_symptoms or unmatched_symptoms"]
    }
  ],
  "disclaimer": "string, standard medical disclaimer"
}
"""


def get_llm_explanation(kg_result: Dict[str, Any], additional_info: Optional[str]) -> Dict[str, Any]:
    if not llm_client:
        raise RuntimeError("Groq client not initialized. Please verify GROQ_API_KEY in Backend/.env")

    user_payload = {
        "kg_candidates": kg_result["top_diseases"],
        "symptoms_not_found_in_knowledge_graph": kg_result["unmatched_symptoms"],
        "patient_provided_context": additional_info or "No additional context provided.",
    }

    # Models to attempt in order of preference
    models_to_try = [GROQ_MODEL] + [m for m in FALLBACK_GROQ_MODELS if m != GROQ_MODEL]
    last_err = None

    for model_name in models_to_try:
        try:
            logger.info("Attempting LLM completion using Groq model: %s", model_name)
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload, indent=2)},
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content.strip()

            try:
                parsed = json.loads(raw_text)
                return parsed
            except json.JSONDecodeError:
                logger.error("LLM did not return valid JSON: %s", raw_text[:500])
                return {
                    "error": "LLM response could not be parsed as JSON.",
                    "raw_response": raw_text,
                }
        except Exception as err:
            logger.warning("Groq model %s failed (%s). Trying fallback...", model_name, err)
            last_err = err

    raise RuntimeError(f"All Groq models failed. Last error: {last_err}")


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(title="ClinAid Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def shutdown_event():
    if driver:
        driver.close()


class DiagnoseRequest(BaseModel):
    symptoms: List[str] = Field(..., min_items=1, description="List of symptom names as reported by the patient")
    additional_info: Optional[str] = Field(
        None, description="Free-text patient context: duration, severity, history, etc."
    )


class DiagnoseResponse(BaseModel):
    kg_evidence: Dict[str, Any]
    llm_diagnosis: Dict[str, Any]
    external_data: Optional[Dict[str, Any]] = None


class ApproveKGRequest(BaseModel):
    items: List[Dict[str, Any]] = Field(
        ...,
        description=(
            'List of approved associations: '
            '[{"symptom": "...", "disease_name": "...", "weight": 0.5, "specificity": 1.0}]'
        ),
    )


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="At least one symptom is required.")

    try:
        kg_result = score_diseases(req.symptoms)
    except Exception as e:
        logger.exception("KG scoring failed")
        raise HTTPException(status_code=500, detail=f"Knowledge graph query failed: {e}")

    # ── External API lookup for unmatched symptoms ──
    external_data = None
    unmatched = kg_result.get("unmatched_symptoms", [])
    if unmatched:
        try:
            external_data = resolve_unmatched_symptoms(
                unmatched_symptoms=unmatched,
                llm_client=llm_client,
                groq_model=GROQ_MODEL,
                fallback_models=FALLBACK_GROQ_MODELS,
            )
        except Exception as e:
            logger.warning("External symptom lookup failed: %s", e)

    # If KG found nothing but external did, build a synthetic evidence object
    # so the LLM still has material to reason over.
    if not kg_result["top_diseases"] and external_data and external_data.get("resolved"):
        # Build synthetic candidates from external data for the LLM
        # Rank 1 gets primary score, Rank 2 & 3 (similar differentials) get calibrated scores
        synthetic_diseases = []
        for entry in external_data["resolved"]:
            for i, d in enumerate(entry.get("diseases", [])[:3]):
                score = round(0.55 - (i * 0.05), 2)
                synthetic_diseases.append({
                    "disease": d["disease_name"],
                    "prior": 0.001,
                    "matched_symptoms": [
                        {
                            "symptom": entry["symptom"],
                            "weight": score,
                            "specificity": 1.0,
                            "contribution": score,
                        }
                    ],
                    "symptom_evidence": score,
                    "missing_symptoms": [],
                    "missing_evidence": 0.0,
                    "final_score": score,
                    "is_related": d.get("is_related", False),
                    "related_to": d.get("related_to"),
                    "clinical_similarity": d.get("clinical_similarity"),
                })
        if synthetic_diseases:
            kg_result["top_diseases"] = synthetic_diseases[:3]

    if not kg_result["top_diseases"]:
        return DiagnoseResponse(
            kg_evidence=kg_result,
            llm_diagnosis={
                "error": "No matching diseases found in the knowledge graph for the given symptoms.",
                "unmatched_symptoms": kg_result["unmatched_symptoms"],
            },
            external_data=external_data,
        )

    try:
        llm_result = get_llm_explanation(kg_result, req.additional_info)
    except Exception as e:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=500, detail=f"LLM reasoning step failed: {e}")

    return DiagnoseResponse(
        kg_evidence=kg_result,
        llm_diagnosis=llm_result,
        external_data=external_data,
    )


@app.post("/approve-kg")
def approve_kg(req: ApproveKGRequest):
    """
    Doctor-approved endpoint: persists externally-resolved symptom→disease
    associations and their related differential diseases into the Neo4j knowledge
    graph after human review, and synchronizes newly approved symptoms with symptoms.json.
    """
    if not req.items:
        raise HTTPException(status_code=400, detail="No items to approve.")

    try:
        result = approve_and_add_to_kg(
            driver=driver,
            database=NEO4J_DATABASE,
            approved_items=req.items,
        )
        return {
            "status": "approved",
            "message": f"Successfully added {result['added_count']} associations to the knowledge graph.",
            "details": result,
            "symptoms_added_to_json": result.get("symptoms_added_to_json", []),
        }
    except Exception as e:
        logger.exception("KG approval write failed")
        raise HTTPException(status_code=500, detail=f"Failed to update knowledge graph: {e}")


@app.get("/health")
def health():
    neo4j_status = "unconfigured"
    if NEO4J_URI and NEO4J_PASSWORD:
        try:
            with driver.session(database=NEO4J_DATABASE) as session:
                session.run("RETURN 1 AS ping").single()
            neo4j_status = "connected"
        except Exception as e:
            neo4j_status = f"error: {e}"

    return {
        "status": "ok",
        "neo4j": neo4j_status,
        "groq_configured": bool(GROQ_API_KEY),
        "groq_model": GROQ_MODEL,
    }
