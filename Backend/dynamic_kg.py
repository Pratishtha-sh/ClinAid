"""
dynamic_kg.py
-------------
External symptom → disease resolution for symptoms NOT found in the local
ClinAid knowledge graph.

Pipeline:
    1.  Convert free-text symptom → HPO term ID via NLM Clinical Tables API.
    2.  Resolve HPO ID → associated diseases via JAX HPO API.
    3.  (Optionally) Fetch co-occurring symptoms for those diseases.
    4.  If external APIs return nothing useful, fall back transparently to the
        LLM for a best-effort answer.
    5.  After a doctor approves the external result, persist the new
        symptom ↔ disease edges into the Neo4j KG.

All public-access, no API key required.
"""

import logging
import json
from pathlib import Path
import httpx
from typing import List, Dict, Any, Optional

logger = logging.getLogger("clinaid.dynamic_kg")

# ---------------------------------------------------------------------------
# NIH / HPO API endpoints (public, unauthenticated)
# ---------------------------------------------------------------------------
NLM_HPO_SEARCH = "https://clinicaltables.nlm.nih.gov/api/hpo/v3/search"
JAX_HPO_DISEASES = "https://phenotype-ontologies-api.jax.org/api/hpo/term/{hpo_id}/diseases"
JAX_DISEASE_PHENOS = "https://phenotype-ontologies-api.jax.org/api/hpo/disease/{disease_id}"

HTTP_TIMEOUT = 10.0  # seconds


# ---------------------------------------------------------------------------
# Step 1 — Symptom text → HPO Term ID
# ---------------------------------------------------------------------------
def symptom_to_hpo_id(symptom: str) -> Optional[Dict[str, str]]:
    """
    Query the NLM Clinical Table Search to convert a symptom string
    into its best-match HPO concept.

    Returns {"hpo_id": "HP:...", "name": "..."} or None.
    """
    try:
        resp = httpx.get(
            NLM_HPO_SEARCH,
            params={"terms": symptom, "df": "id,name"},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        # Response format: [total, [ids...], null, [[id, name], ...]]
        if not data or len(data) < 4 or not data[3]:
            return None

        best = data[3][0]  # first (highest-ranked) match
        return {"hpo_id": best[0], "name": best[1]}
    except Exception as e:
        logger.warning("NLM HPO lookup failed for '%s': %s", symptom, e)
        return None


# ---------------------------------------------------------------------------
# Step 2 — HPO Term ID → Associated Diseases
# ---------------------------------------------------------------------------
def hpo_id_to_diseases(hpo_id: str, max_diseases: int = 5) -> List[Dict[str, str]]:
    """
    Query the JAX HPO API for diseases associated with a given HPO term.

    Returns a list of {"disease_id": "OMIM:...", "disease_name": "..."}.
    """
    try:
        url = JAX_HPO_DISEASES.format(hpo_id=hpo_id)
        resp = httpx.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        diseases = []
        items = payload if isinstance(payload, list) else payload.get("diseases", payload.get("associations", []))
        for entry in items[:max_diseases]:
            disease_id = entry.get("diseaseId") or entry.get("disease_id") or entry.get("id", "")
            disease_name = entry.get("diseaseName") or entry.get("disease_name") or entry.get("name", "")
            if disease_name:
                diseases.append({"disease_id": disease_id, "disease_name": disease_name})

        return diseases
    except Exception as e:
        logger.warning("JAX disease lookup failed for '%s': %s", hpo_id, e)
        return []


# ---------------------------------------------------------------------------
# Step 3 — Disease ID → co-occurring symptoms (optional enrichment)
# ---------------------------------------------------------------------------
def disease_to_symptoms(disease_id: str) -> List[str]:
    """
    Fetch additional HPO phenotype terms associated with a disease.
    Useful for enriching the KG after doctor approval.
    """
    try:
        url = JAX_DISEASE_PHENOS.format(disease_id=disease_id)
        resp = httpx.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        symptoms = []
        # The response may be a list or an object with a nested list
        items = payload if isinstance(payload, list) else payload.get("phenotypes", payload.get("associations", []))
        for entry in items:
            name = entry.get("name") or entry.get("label") or entry.get("ontologyTerm", {}).get("name", "")
            if name:
                symptoms.append(name)
        return symptoms
    except Exception as e:
        logger.warning("JAX phenotype lookup failed for '%s': %s", disease_id, e)
        return []


# ---------------------------------------------------------------------------
# Full external lookup pipeline (called from diagnosis_api.py)
# ---------------------------------------------------------------------------
def lookup_external(
    unmatched_symptoms: List[str],
) -> Dict[str, Any]:
    """
    For each unmatched symptom, attempt the NLM → JAX pipeline to discover
    plausible diseases from external ontology data.

    Returns:
        {
            "source": "external_api",
            "resolved": [
                {
                    "symptom": "...",
                    "hpo": {"hpo_id": "HP:...", "name": "..."},
                    "diseases": [{"disease_id": "...", "disease_name": "..."}, ...]
                }, ...
            ],
            "still_unresolved": ["symptom_x", ...]
        }
    """
    resolved = []
    still_unresolved = []

    for symptom in unmatched_symptoms:
        hpo = symptom_to_hpo_id(symptom)
        if not hpo:
            still_unresolved.append(symptom)
            continue

        diseases = hpo_id_to_diseases(hpo["hpo_id"])
        if diseases:
            resolved.append({
                "symptom": symptom,
                "hpo": hpo,
                "diseases": diseases,
            })
        else:
            still_unresolved.append(symptom)

    return {
        "source": "external_api",
        "resolved": resolved,
        "still_unresolved": still_unresolved,
    }


# ---------------------------------------------------------------------------
# LLM-based fallback for symptoms the external APIs couldn't resolve.
# This is called internally and the source is never exposed to the user.
# ---------------------------------------------------------------------------
def _llm_fallback_diagnosis(
    symptoms: List[str],
    llm_client,
    groq_model: str,
    fallback_models: List[str],
) -> Optional[Dict[str, Any]]:
    """
    When external APIs fail, quietly ask the LLM for a best-effort mapping
    of symptoms → possible diseases. The output is shaped identically to the
    external-API result so the downstream pipeline is source-agnostic.
    """
    if not llm_client or not symptoms:
        return None

    prompt = (
        "You are a medical knowledge assistant. Given the following patient symptoms "
        "that could not be found in our clinical knowledge graph, suggest the most likely "
        "diseases associated with each symptom. Respond ONLY with valid JSON matching this schema:\n"
        '{\n'
        '  "resolved": [\n'
        '    {\n'
        '      "symptom": "symptom name",\n'
        '      "diseases": [\n'
        '        {"disease_name": "Disease A"},\n'
        '        {"disease_name": "Disease B"}\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        f"Symptoms: {json.dumps(symptoms)}"
    )

    models_to_try = [groq_model] + [m for m in fallback_models if m != groq_model]
    for model_name in models_to_try:
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a clinical knowledge mapping assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            # Normalize output to match external_api shape
            result_resolved = []
            for item in parsed.get("resolved", []):
                diseases = []
                for d in item.get("diseases", []):
                    diseases.append({
                        "disease_id": "",
                        "disease_name": d.get("disease_name", d.get("name", "")),
                    })
                result_resolved.append({
                    "symptom": item.get("symptom", ""),
                    "hpo": None,
                    "diseases": diseases,
                })

            return {
                "source": "external_api",  # intentionally labelled the same
                "resolved": result_resolved,
                "still_unresolved": [],
            }
        except Exception as e:
            logger.warning("LLM fallback model %s failed: %s", model_name, e)

    return None


# ---------------------------------------------------------------------------
# Symptoms JSON Synchronization
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
SYMPTOMS_JSON_PATHS = [
    PROJECT_ROOT / "Frontend" / "src" / "data" / "symptoms.json",
    PROJECT_ROOT / "Frontend" / "dist" / "data" / "symptoms.json",
]


def add_symptoms_to_json(symptoms: List[str]) -> List[str]:
    """
    Appends newly approved symptoms to Frontend/src/data/symptoms.json
    (and Frontend/dist/data/symptoms.json if present), maintaining a clean,
    lowercase, alphabetically sorted JSON list without duplicates.

    Returns the list of newly added symptoms.
    """
    if not symptoms:
        return []

    target_file = SYMPTOMS_JSON_PATHS[0]
    if not target_file.is_file():
        logger.warning("symptoms.json not found at %s", target_file)
        return []

    newly_added = []
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            current_list = json.load(f)
        if not isinstance(current_list, list):
            current_list = []

        existing_set = {s.strip().lower() for s in current_list if isinstance(s, str)}

        for sym in symptoms:
            clean_sym = sym.strip().lower()
            if clean_sym and clean_sym not in existing_set:
                current_list.append(clean_sym)
                existing_set.add(clean_sym)
                newly_added.append(clean_sym)

        if newly_added:
            # Sort alphabetically
            sorted_list = sorted(list(set(s.strip().lower() for s in current_list if s and s.strip())))
            for p in SYMPTOMS_JSON_PATHS:
                if p.parent.is_dir():
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(sorted_list, f, indent=2, ensure_ascii=False)
            logger.info("Successfully added %d symptom(s) to symptoms.json: %s", len(newly_added), newly_added)

    except Exception as e:
        logger.error("Failed to update symptoms.json: %s", e)

    return newly_added


# ---------------------------------------------------------------------------
# LLM Similar Disease Retrieval (for Rank 2 & 3 differentials)
# ---------------------------------------------------------------------------
def get_similar_diseases_to_rank1(
    primary_disease: str,
    symptom: str,
    llm_client,
    groq_model: str,
    fallback_models: List[str] = None,
    count: int = 2,
) -> List[Dict[str, Any]]:
    """
    Uses the LLM to identify clinically plausible, standard differential diagnoses
    that are closely similar to the primary (Rank 1) candidate condition and also present
    with the specified symptom. Replaces irrelevant outlier conditions with coherent differentials.
    """
    if not llm_client or not primary_disease:
        return []

    prompt = (
        f"You are a clinical differential diagnosis consultant.\n"
        f"A patient presents with the symptom: '{symptom}'.\n"
        f"The primary candidate disease (Rank 1) identified is: '{primary_disease}'.\n\n"
        f"Provide exactly {count} distinct, clinically plausible differential diagnoses or related diseases that:\n"
        f"1. Share clinical presentation, organ system, or pathophysiological category with '{primary_disease}'.\n"
        f"2. Are realistic, common clinical conditions that standard medical practice would consider alongside '{primary_disease}' for '{symptom}'.\n"
        f"3. Avoid obscure, ultra-rare genetic syndromes; provide common, well-recognized clinical differentials.\n\n"
        "Respond ONLY with valid JSON matching this schema:\n"
        "{\n"
        '  "related_diseases": [\n'
        '    {\n'
        '      "disease_name": "Disease Name",\n'
        '      "clinical_similarity": "Brief 1-sentence clinical explanation of why it is a close differential to Rank 1"\n'
        '    }\n'
        "  ]\n"
        "}"
    )

    models_to_try = [groq_model] + [m for m in (fallback_models or []) if m != groq_model]
    for model_name in models_to_try:
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a senior clinical consultant in differential diagnosis."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            items = parsed.get("related_diseases", parsed.get("diseases", []))
            results = []
            for item in items[:count]:
                dname = item.get("disease_name") or item.get("name") or ""
                if dname and dname.strip().lower() != primary_disease.strip().lower():
                    results.append({
                        "disease_id": "",
                        "disease_name": dname.strip(),
                        "clinical_similarity": item.get("clinical_similarity", f"Related differential to {primary_disease}"),
                        "is_related": True,
                        "related_to": primary_disease,
                    })
            if results:
                return results
        except Exception as e:
            logger.warning("LLM model %s failed in get_similar_diseases_to_rank1: %s", model_name, e)

    return []


def resolve_unmatched_symptoms(
    unmatched_symptoms: List[str],
    llm_client=None,
    groq_model: str = "",
    fallback_models: List[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Main entry point: tries external ontology lookup first, falls back to LLM,
    and then uses the LLM to generate clinically coherent differential diagnoses
    similar to the Rank 1 disease to replace irrelevant Rank 2 & 3 diseases.
    """
    if not unmatched_symptoms:
        return None

    external = lookup_external(unmatched_symptoms)

    # If there are still unresolved symptoms, use the LLM to fill the gaps
    if external["still_unresolved"] and llm_client:
        llm_result = _llm_fallback_diagnosis(
            external["still_unresolved"],
            llm_client,
            groq_model,
            fallback_models or [],
        )
        if llm_result and llm_result.get("resolved"):
            external["resolved"].extend(llm_result["resolved"])
            external["still_unresolved"] = []

    if not external["resolved"]:
        return None

    # For each resolved symptom, ensure Rank 1 is clear, and query LLM for
    # clinically similar differential diagnoses for Rank 2 and Rank 3.
    if llm_client:
        for entry in external["resolved"]:
            diseases = entry.get("diseases", [])
            if not diseases:
                continue

            # The top disease is Rank 1 (Primary Condition)
            primary_disease_obj = diseases[0]
            primary_disease_name = primary_disease_obj.get("disease_name", "")
            primary_disease_obj["is_primary"] = True
            primary_disease_obj["rank"] = 1

            # Query LLM for 2 similar/related differential diseases to Rank 1
            similar = get_similar_diseases_to_rank1(
                primary_disease=primary_disease_name,
                symptom=entry["symptom"],
                llm_client=llm_client,
                groq_model=groq_model,
                fallback_models=fallback_models,
                count=2,
            )

            if similar:
                for idx, sim in enumerate(similar):
                    sim["rank"] = idx + 2
                # Replace diseases list so Rank 1, 2, 3 are coherent
                entry["diseases"] = [primary_disease_obj] + similar
                entry["related_diseases"] = [s["disease_name"] for s in similar]
            else:
                # Fallback if similar query couldn't complete
                for idx, d in enumerate(entry["diseases"]):
                    d["rank"] = idx + 1
                    if idx > 0:
                        d["is_related"] = True
                        d["related_to"] = primary_disease_name

    return external


# ---------------------------------------------------------------------------
# Doctor-approval: persist external findings into the Neo4j KG
# ---------------------------------------------------------------------------
def approve_and_add_to_kg(
    driver,
    database: str,
    approved_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    After a doctor reviews and approves externally-resolved symptom→disease
    associations, write them and their related differential diseases into the
    Neo4j knowledge graph, and update symptoms.json.

    approved_items: list of {
        "symptom": "...",
        "disease_name": "...",
        "weight": 0.5,        # optional, default 0.5
        "specificity": 1.0,   # optional, default 1.0
        "related_diseases": ["Disease B", "Disease C"],  # optional
        "related_to": "Disease A"                        # optional
    }

    Returns summary of what was added.
    """
    added = []
    symptoms_to_add = set()

    edge_query = """
    MERGE (s:Symptom {name: $symptom})
    ON CREATE SET s.specificity = $specificity, s.source = 'external_api'
    MERGE (d:Disease {name: $disease})
    ON CREATE SET d.prior = 0.001, d.occurrence_count = 1, d.source = 'external_api'
    MERGE (d)-[r:HAS_SYMPTOM]->(s)
    ON CREATE SET r.weight = $weight, r.credit_sum = $weight, r.source = 'external_api'
    ON MATCH SET  r.weight = CASE WHEN r.source = 'external_api'
                                   THEN $weight ELSE r.weight END
    RETURN d.name AS disease, s.name AS symptom
    """

    rel_query = """
    MERGE (d1:Disease {name: $primary_disease})
    MERGE (d2:Disease {name: $related_disease})
    ON CREATE SET d2.prior = 0.001, d2.occurrence_count = 1, d2.source = 'llm_related'
    MERGE (d1)-[rel:RELATED_TO]-(d2)
    ON CREATE SET rel.source = 'llm_differential'
    MERGE (s:Symptom {name: $symptom})
    MERGE (d2)-[r:HAS_SYMPTOM]->(s)
    ON CREATE SET r.weight = $rel_weight, r.credit_sum = $rel_weight, r.source = 'llm_related'
    RETURN d1.name AS primary, d2.name AS related, s.name AS symptom
    """

    with driver.session(database=database) as session:
        for item in approved_items:
            symptom_name = item.get("symptom")
            disease_name = item.get("disease_name")
            if not symptom_name or not disease_name:
                continue

            symptoms_to_add.add(symptom_name)

            # 1. Add primary/approved disease -> symptom edge
            try:
                result = session.run(
                    edge_query,
                    symptom=symptom_name,
                    disease=disease_name,
                    weight=item.get("weight", 0.5),
                    specificity=item.get("specificity", 1.0),
                )
                record = result.single()
                if record:
                    added.append({
                        "disease": record["disease"],
                        "symptom": record["symptom"],
                        "is_related": item.get("is_related", False),
                    })
            except Exception as e:
                logger.error(
                    "Failed to add %s → %s to KG: %s",
                    symptom_name, disease_name, e,
                )

            # 2. Add related differential diseases if provided
            related_list = item.get("related_diseases", [])
            for rel_d in related_list:
                if rel_d and rel_d.strip().lower() != disease_name.strip().lower():
                    try:
                        session.run(
                            rel_query,
                            primary_disease=disease_name,
                            related_disease=rel_d.strip(),
                            symptom=symptom_name,
                            rel_weight=item.get("rel_weight", 0.45),
                        )
                        added.append({
                            "disease": rel_d.strip(),
                            "symptom": symptom_name,
                            "is_related": True,
                            "related_to": disease_name,
                        })
                    except Exception as e:
                        logger.error(
                            "Failed to add related disease %s for %s to KG: %s",
                            rel_d, disease_name, e,
                        )

            # 3. If item itself specifies related_to, ensure RELATED_TO link exists
            related_to = item.get("related_to")
            if related_to and related_to.strip().lower() != disease_name.strip().lower():
                try:
                    session.run(
                        """
                        MERGE (d1:Disease {name: $p})
                        MERGE (d2:Disease {name: $r})
                        MERGE (d1)-[rel:RELATED_TO]-(d2)
                        ON CREATE SET rel.source = 'llm_differential'
                        """,
                        p=related_to.strip(),
                        r=disease_name.strip(),
                    )
                except Exception as e:
                    logger.error(
                        "Failed to create RELATED_TO edge %s - %s: %s",
                        related_to, disease_name, e,
                    )

    # Automatically add new symptom(s) to symptoms.json
    added_symptoms_json = add_symptoms_to_json(list(symptoms_to_add))

    return {
        "added_count": len(added),
        "edges": added,
        "symptoms_added_to_json": added_symptoms_json,
    }