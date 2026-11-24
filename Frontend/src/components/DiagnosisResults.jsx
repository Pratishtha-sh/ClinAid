import React, { useState, useEffect } from 'react'
import {
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Activity,
  Globe,
  ShieldCheck,
  Loader2,
  CheckCircle,
  XCircle,
  Sparkles,
} from 'lucide-react'
import EvidenceAnimation from './EvidenceAnimation'
import { approveKG } from '../services/api'

export default function DiagnosisResults({ result, loading, externalData }) {
  const [approvalState, setApprovalState] = useState({})
  // Track per-item checkbox state: { "symptom||disease": true/false }
  const [selectedItems, setSelectedItems] = useState({})
  const [approving, setApproving] = useState(false)
  const [approvalResult, setApprovalResult] = useState(null)

  // Build external resolved items for the approval panel
  const externalResolved = externalData?.resolved || []
  const hasExternalData = externalResolved.length > 0

  // Pre-select all resolved disease items by default so the clinician can approve them together
  // MUST be called before any early return to adhere to React Rules of Hooks!
  useEffect(() => {
    if (externalResolved && externalResolved.length > 0) {
      const initialSelected = {}
      for (const entry of externalResolved) {
        for (const d of entry.diseases || []) {
          const key = `${entry.symptom}||${d.disease_name}`
          initialSelected[key] = true
        }
      }
      setSelectedItems((prev) => ({ ...initialSelected, ...prev }))
    }
  }, [externalData])

  if (loading) {
    return (
      <div className="diagnosis-loading-card">
        <div className="pulse-loader" />
        <div>
          <h3 style={{ fontSize: '1.3rem', marginBottom: '8px', color: '#0b2545' }}>
            Triaging Patient Presentation
          </h3>
          <p style={{ color: '#64748b', fontSize: '0.9rem' }}>
            Traversing knowledge graph & generating grounded clinical reasoning...
          </p>
        </div>
        <div className="loading-steps">
          <div className="step-item active">
            <CheckCircle2 size={16} color="#0b2545" />
            <span>Querying clinical knowledge graph for disease-symptom associations...</span>
          </div>
          <div className="step-item active">
            <CheckCircle2 size={16} color="#0b2545" />
            <span>Computing Symptom Specificity & Evidence Association Weights...</span>
          </div>
          <div className="step-item active">
            <CheckCircle2 size={16} color="#0b2545" />
            <span>Applying calibrated missing expected symptom penalty...</span>
          </div>
          <div className="step-item active">
            <Activity size={16} color="#10b981" />
            <span>Synthesizing evidence-based diagnostic explanations...</span>
          </div>
        </div>
      </div>
    )
  }

  if (!result) return null

  const kg_evidence = result?.kg_evidence || {}
  const llm_diagnosis = result?.llm_diagnosis || {}
  const assessments = Array.isArray(llm_diagnosis?.assessment) ? llm_diagnosis.assessment : []
  const topKgDiseases = Array.isArray(kg_evidence?.top_diseases) ? kg_evidence.top_diseases : []
  const inputSymptoms =
    result.input_symptoms ||
    result.inputSymptoms ||
    kg_evidence?.input_symptoms ||
    []

  const toggleItem = (key) => {
    setSelectedItems((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const handleApprove = async () => {
    const itemsToApprove = []
    for (const entry of externalResolved) {
      const allRelatedDiseases = (entry.diseases || [])
        .filter((d) => d.is_related || (d.rank && d.rank > 1))
        .map((d) => d.disease_name)

      for (const d of entry.diseases || []) {
        const key = `${entry.symptom}||${d.disease_name}`
        if (selectedItems[key]) {
          itemsToApprove.push({
            symptom: entry.symptom,
            disease_name: d.disease_name,
            weight: d.rank === 1 ? 0.55 : 0.45,
            specificity: 1.0,
            is_related: !!d.is_related,
            related_to: d.related_to || null,
            related_diseases: allRelatedDiseases,
          })
        }
      }
    }

    if (itemsToApprove.length === 0) return

    setApproving(true)
    setApprovalResult(null)
    try {
      const res = await approveKG(itemsToApprove)
      const extraMsg = res.symptoms_added_to_json?.length
        ? ` (Added "${res.symptoms_added_to_json.join(', ')}" to symptoms.json)`
        : ''
      setApprovalResult({
        success: true,
        message: (res.message || 'Associations added to Knowledge Graph.') + extraMsg,
      })

      // Mark all approved items
      const newState = {}
      for (const item of itemsToApprove) {
        newState[`${item.symptom}||${item.disease_name}`] = 'approved'
      }
      setApprovalState((prev) => ({ ...prev, ...newState }))

      // Notify the app of newly approved symptoms so the diagnosis form updates dynamically
      window.dispatchEvent(
        new CustomEvent('clinaid:symptom-approved', {
          detail: {
            symptoms: itemsToApprove.map((i) => i.symptom),
            added_to_json: res.symptoms_added_to_json || [],
          },
        })
      )
    } catch (err) {
      setApprovalResult({ success: false, message: err.message || 'Approval failed.' })
    } finally {
      setApproving(false)
    }
  }

  const selectedCount = Object.values(selectedItems).filter(Boolean).length

  return (
    <div id="results-area" className="results-container">
      {/* ── External Data Banner ── */}
      {hasExternalData && (
        <div className="external-data-banner">
          <div className="external-banner-header">
            <div className="external-banner-icon-wrap">
              <Globe size={22} color="#ffffff" />
            </div>
            <div>
              <h3 className="external-banner-title">
                External Symptom Resolution &amp; Differential Expansion
              </h3>
              <p className="external-banner-subtitle">
                Some symptoms were not found in the local knowledge graph. The primary candidate was
                resolved and <strong>clinically similar differential diagnoses</strong> were identified
                via LLM to form a coherent top-3 differential panel.
              </p>
            </div>
          </div>

          <div className="external-resolved-list">
            {externalResolved.map((entry, idx) => (
              <div key={idx} className="external-resolved-item">
                <div className="external-symptom-label">
                  <span className="external-symptom-tag">{entry.symptom}</span>
                  {entry.hpo && (
                    <span className="external-hpo-id">{entry.hpo.hpo_id}</span>
                  )}
                </div>
                <div className="external-diseases-list">
                  {(entry.diseases || []).map((d, di) => {
                    const key = `${entry.symptom}||${d.disease_name}`
                    const isApproved = approvalState[key] === 'approved'
                    const isPrimary = d.rank === 1 || d.is_primary || di === 0
                    return (
                      <label
                        key={di}
                        className={`external-disease-checkbox ${isApproved ? 'approved' : ''} ${isPrimary ? 'primary-card' : 'related-card'}`}
                      >
                        <input
                          type="checkbox"
                          checked={!!selectedItems[key] || isApproved}
                          disabled={isApproved}
                          onChange={() => toggleItem(key)}
                        />
                        <div className="external-disease-content">
                          <div className="external-disease-row">
                            <span className="external-disease-name">{d.disease_name}</span>
                            {isPrimary ? (
                              <span className="badge-rank-primary">Rank #1 Primary</span>
                            ) : (
                              <span className="badge-rank-related">
                                <Sparkles size={11} style={{ marginRight: 3, verticalAlign: 'middle' }} />
                                Rank #{d.rank || di + 1} Similar Differential
                              </span>
                            )}
                            {d.disease_id && (
                              <span className="external-disease-id">{d.disease_id}</span>
                            )}
                          </div>
                          {d.clinical_similarity && (
                            <div className="external-disease-similarity">
                              {d.clinical_similarity}
                            </div>
                          )}
                        </div>
                        {isApproved && (
                          <CheckCircle size={16} className="approved-icon" />
                        )}
                      </label>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>

          {/* Doctor Approval Button */}
          <div className="external-approval-bar">
            <div className="approval-info">
              <ShieldCheck size={18} />
              <span>
                Select associations to approve and add to the Knowledge Graph.
                <strong> {selectedCount}</strong> selected.
              </span>
            </div>
            <button
              className="btn-approve-kg"
              disabled={selectedCount === 0 || approving}
              onClick={handleApprove}
            >
              {approving ? (
                <>
                  <Loader2 size={16} className="spin-icon" />
                  <span>Adding to KG...</span>
                </>
              ) : (
                <>
                  <ShieldCheck size={16} />
                  <span>Approve & Add to KG</span>
                </>
              )}
            </button>
          </div>

          {approvalResult && (
            <div className={`approval-result-banner ${approvalResult.success ? 'success' : 'error'}`}>
              {approvalResult.success ? (
                <CheckCircle size={16} color="#059669" />
              ) : (
                <XCircle size={16} color="#dc2626" />
              )}
              <span>{approvalResult.message}</span>
            </div>
          )}
        </div>
      )}

      {/* Header */}
      <div className="results-header">
        <div>
          <div
            className="section-badge"
            style={{
              background: '#f0fdf4',
              borderColor: '#bbf7d0',
              color: '#166534',
            }}
          >
            <Activity size={14} color="#10b981" />
            <span>Differential Diagnostic Assessment</span>
          </div>
          <h2 style={{ fontSize: '2.1rem', fontWeight: 800, color: '#0b2545' }}>
            Top-3 Differential Diagnoses
          </h2>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span style={{ fontSize: '0.82rem', color: '#64748b' }}>Diagnostic Engine:</span>
          <p style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0b2545' }}>
            Clinical Knowledge Graph & Reasoning Pipeline
          </p>
        </div>
      </div>

      {/* Top 3 Disease Cards */}
      <div className="diseases-grid">
        {assessments.length > 0
          ? assessments.map((item, idx) => {
              const rank = item.rank || idx + 1
              const confClass =
                item.confidence_level?.toLowerCase() === 'high'
                  ? 'confidence-high'
                  : item.confidence_level?.toLowerCase() === 'moderate'
                  ? 'confidence-moderate'
                  : 'confidence-low'

              return (
                <div key={idx} className={`disease-card rank-${rank}`}>
                  <div className="disease-card-header">
                    <div>
                      <span className="rank-badge">Rank #{rank}</span>
                      <h3 className="disease-name">{item.disease}</h3>
                    </div>
                    <span className={`confidence-pill ${confClass}`}>
                      {item.confidence_level || 'Evaluated'}
                    </span>
                  </div>

                  {/* Clinical Rationale */}
                  <div className="reasoning-text">
                    <p>{item.reasoning}</p>
                  </div>

                  {/* Supporting Evidence */}
                  <div>
                    <div className="evidence-section-title">
                      <CheckCircle2 size={14} color="#10b981" />
                      <span>Supporting Evidence</span>
                    </div>
                    <ul className="evidence-list">
                      {item.supporting_evidence?.length > 0 ? (
                        item.supporting_evidence.map((ev, i) => (
                          <li key={i} className="evidence-item positive">
                            <CheckCircle2 size={15} />
                            <span>{ev}</span>
                          </li>
                        ))
                      ) : (
                        <li className="evidence-item" style={{ color: '#64748b' }}>
                          Evidence correlated from knowledge graph weights.
                        </li>
                      )}
                    </ul>
                  </div>

                  {/* Diagnostic Gaps / Missing Expected Symptoms */}
                  <div>
                    <div className="evidence-section-title">
                      <AlertTriangle size={14} color="#b45309" />
                      <span>Diagnostic Gaps / Absent Symptoms</span>
                    </div>
                    <ul className="evidence-list">
                      {item.concerns_or_gaps?.length > 0 ? (
                        item.concerns_or_gaps.map((gap, i) => (
                          <li key={i} className="evidence-item warning">
                            <AlertTriangle size={15} />
                            <span>{gap}</span>
                          </li>
                        ))
                      ) : (
                        <li className="evidence-item" style={{ color: '#64748b' }}>
                          No major absent expected symptoms flagged.
                        </li>
                      )}
                    </ul>
                  </div>
                </div>
              )
            })
          : /* Fallback if LLM payload wasn't parsed as array */
            topKgDiseases.map((kgItem, idx) => (
              <div key={idx} className={`disease-card rank-${idx + 1}`}>
                <span className="rank-badge">Rank #{idx + 1}</span>
                <h3 className="disease-name">{kgItem.disease}</h3>
                <p style={{ color: '#64748b' }}>
                  Score: <strong>{kgItem.final_score}</strong>
                </p>
              </div>
            ))}
      </div>

      {/* Interactive Evidence Propagation & Scoring Animation */}
      <EvidenceAnimation
        inputSymptoms={inputSymptoms}
        candidates={topKgDiseases}
      />

      {/* Medical Disclaimer */}
      <div className="disclaimer-banner">
        <ShieldAlert size={24} color="#b91c1c" style={{ flexShrink: 0 }} />
        <div>
          <strong>Medical Disclaimer:</strong>{' '}
          {llm_diagnosis?.disclaimer ||
            'ClinAid is a clinical decision-support and educational tool designed for healthcare professionals. It does not replace licensed medical judgment, laboratory confirmation, or emergency medical evaluation.'}
        </div>
      </div>
    </div>
  )
}
