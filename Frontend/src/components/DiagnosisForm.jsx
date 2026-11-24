import React, { useState, useMemo, useRef, useEffect } from 'react'
import { Search, X, Plus, Activity, FileText, Clock, AlertCircle, Check, Globe, Sparkles } from 'lucide-react'
import allSymptoms from '../data/symptoms.json'

// Build a fast lookup set (lowercased) for checking known vs new symptoms
const knownSymptomsSet = new Set(allSymptoms.map((s) => s.toLowerCase()))

export default function DiagnosisForm({ onSubmit, loading, error }) {
  const [symptomsList, setSymptomsList] = useState(allSymptoms)
  const [query, setQuery] = useState('')
  const [selectedSymptoms, setSelectedSymptoms] = useState([
    'anxiety and nervousness',
    'shortness of breath',
    'sharp chest pain',
    'palpitations',
  ])
  const [duration, setDuration] = useState('Acute (past 4–6 hours)')
  const [severity, setSeverity] = useState('Moderate')
  const [additionalNotes, setAdditionalNotes] = useState(
    'Patient reports rapid heartbeat and tightness across upper chest while at rest. No prior history of hypertension.'
  )
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const dropdownRef = useRef(null)

  useEffect(() => {
    const handleSymptomApproved = (e) => {
      const approved = e.detail?.symptoms || []
      if (!approved.length) return
      setSymptomsList((prev) => {
        const next = [...prev]
        for (const sym of approved) {
          const lower = sym.toLowerCase().trim()
          if (!next.some((s) => s.toLowerCase() === lower)) {
            next.push(lower)
            knownSymptomsSet.add(lower)
          }
        }
        return next.sort()
      })
    }
    window.addEventListener('clinaid:symptom-approved', handleSymptomApproved)
    return () => window.removeEventListener('clinaid:symptom-approved', handleSymptomApproved)
  }, [])

  const isKnownSymptom = (sym) =>
    knownSymptomsSet.has(sym.toLowerCase()) ||
    symptomsList.some((s) => s.toLowerCase() === sym.toLowerCase())

  const filteredSymptoms = useMemo(() => {
    if (!query.trim()) return []
    const q = query.toLowerCase().trim()
    return symptomsList.filter((s) => s.toLowerCase().includes(q)).slice(0, 10)
  }, [query, symptomsList])

  // Check if the current query is an exact match to an existing symptom
  const queryTrimmed = query.trim()
  const queryLower = queryTrimmed.toLowerCase()
  const isExactMatch = queryTrimmed && symptomsList.some((s) => s.toLowerCase() === queryLower)
  const isAlreadySelected = queryTrimmed && selectedSymptoms.some((s) => s.toLowerCase() === queryLower)
  const showAddCustom = queryTrimmed.length >= 2 && !isExactMatch && !isAlreadySelected

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleAddSymptom = (sym) => {
    const normalized = sym.trim()
    if (!normalized) return
    if (!selectedSymptoms.some((s) => s.toLowerCase() === normalized.toLowerCase())) {
      setSelectedSymptoms((prev) => [...prev, normalized])
    }
    setQuery('')
    setIsDropdownOpen(false)
  }

  const handleAddCustomSymptom = () => {
    handleAddSymptom(queryTrimmed)
  }

  const handleRemoveSymptom = (sym) => {
    setSelectedSymptoms((prev) => prev.filter((s) => s !== sym))
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      if (filteredSymptoms.length > 0 && !showAddCustom) {
        // Add the first matched known symptom
        const first = filteredSymptoms[0]
        if (!selectedSymptoms.includes(first)) handleAddSymptom(first)
      } else if (showAddCustom) {
        // Add as custom symptom
        handleAddCustomSymptom()
      }
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (selectedSymptoms.length === 0) return
    const fullContext = `[Duration/Onset]: ${duration}. [Severity]: ${severity}. [Notes & History]: ${additionalNotes}`
    onSubmit({ symptoms: selectedSymptoms, additional_info: fullContext })
  }

  const knownCount = selectedSymptoms.filter((s) => isKnownSymptom(s)).length
  const newCount = selectedSymptoms.length - knownCount

  return (
    <section id="diagnosis-section" className="diagnosis-section">
      <div className="section-header">
        <p className="section-eyebrow">Physician Input Interface</p>
        <h2 className="section-title">Clinical Diagnostic Assessment</h2>
        <p className="section-subtitle">
          Select observed patient symptoms and provide relevant clinical history
          for structured, transparent differential diagnosis ranking.
        </p>
      </div>

      <form className="form-card" onSubmit={handleSubmit}>

        {/* ── Step 1: Symptom Search ── */}
        <div className="form-section">
          <div className="form-section-header">
            <div className="form-step-badge">01</div>
            <div>
              <div className="form-section-title">Select Patient Symptoms</div>
              <div className="form-section-desc">
                Search from 377 recognized clinical entities or type a new symptom
              </div>
            </div>
          </div>

          <div className="symptom-search-box" ref={dropdownRef}>
            <div className="input-label-row">
              <label className="input-label" htmlFor="symptom-input">
                Search &amp; Add Symptoms
              </label>
              <span className="input-count-badge">
                {selectedSymptoms.length} Selected
                {newCount > 0 && (
                  <span className="new-count-indicator"> · {newCount} new</span>
                )}
              </span>
            </div>

            <div className="search-field-wrap">
              <Search className="search-icon" size={17} />
              <input
                id="symptom-input"
                type="text"
                className="search-input"
                placeholder="e.g. chest pain, dyspnea, nausea — or type any new symptom..."
                value={query}
                onChange={(e) => { setQuery(e.target.value); setIsDropdownOpen(true) }}
                onFocus={() => setIsDropdownOpen(true)}
                onKeyDown={handleKeyDown}
                autoComplete="off"
              />
            </div>

            {isDropdownOpen && (filteredSymptoms.length > 0 || showAddCustom) && (
              <div className="symptom-dropdown">
                {/* Known symptom matches */}
                {filteredSymptoms.map((sym) => {
                  const isSelected = selectedSymptoms.includes(sym)
                  return (
                    <div
                      key={sym}
                      className={`dropdown-item ${isSelected ? 'already-selected' : ''}`}
                      onClick={() => !isSelected && handleAddSymptom(sym)}
                    >
                      <span>{sym}</span>
                      {isSelected
                        ? <Check size={14} color="#0b2545" />
                        : <Plus size={14} color="#0b2545" />
                      }
                    </div>
                  )
                })}

                {/* Add as new/custom symptom option */}
                {showAddCustom && (
                  <>
                    {filteredSymptoms.length > 0 && (
                      <div className="dropdown-divider" />
                    )}
                    <div
                      className="dropdown-item dropdown-item-new"
                      onClick={handleAddCustomSymptom}
                    >
                      <div className="new-symptom-label">
                        <Globe size={14} className="new-symptom-icon" />
                        <span>
                          Add "<strong>{queryTrimmed}</strong>" as new symptom
                        </span>
                      </div>
                      <span className="new-symptom-badge">External Lookup</span>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="selected-symptoms-tray">
            {selectedSymptoms.length === 0 ? (
              <span className="empty-tray-notice">No symptoms added yet. Search above to begin.</span>
            ) : (
              selectedSymptoms.map((sym) => {
                const isNew = !isKnownSymptom(sym)
                return (
                  <span key={sym} className={`symptom-tag ${isNew ? 'symptom-tag-new' : ''}`}>
                    {isNew && <Globe size={12} className="tag-new-icon" />}
                    <span>{sym}</span>
                    {isNew && <span className="tag-new-label">NEW</span>}
                    <button type="button" className="tag-remove-btn" onClick={() => handleRemoveSymptom(sym)}>
                      <X size={13} />
                    </button>
                  </span>
                )
              })
            )}
          </div>

          {/* Hint when new symptoms are present */}
          {newCount > 0 && (
            <div className="new-symptoms-hint">
              <Sparkles size={14} />
              <span>
                <strong>{newCount} new symptom{newCount > 1 ? 's' : ''}</strong> not in local knowledge graph —
                will be resolved via <strong>NIH external APIs</strong> during analysis.
              </span>
            </div>
          )}
        </div>

        {/* ── Step 2: Clinical Context ── */}
        <div className="form-section">
          <div className="form-section-header">
            <div className="form-step-badge">02</div>
            <div>
              <div className="form-section-title">Clinical Context</div>
              <div className="form-section-desc">Onset duration and severity classification</div>
            </div>
          </div>

          <div className="context-row">
            <div className="form-field">
              <label className="field-label" htmlFor="duration-input">
                <Clock size={14} style={{ display: 'inline', marginRight: '6px', verticalAlign: '-2px' }} />
                Onset &amp; Symptom Duration
              </label>
              <input
                id="duration-input"
                type="text"
                className="text-input"
                value={duration}
                onChange={(e) => setDuration(e.target.value)}
                placeholder="e.g. 2 days, sudden onset, recurrent for 3 weeks"
              />
            </div>

            <div className="form-field">
              <label className="field-label" htmlFor="severity-select">
                <Activity size={14} style={{ display: 'inline', marginRight: '6px', verticalAlign: '-2px' }} />
                Clinical Severity
              </label>
              <select
                id="severity-select"
                className="select-input"
                value={severity}
                onChange={(e) => setSeverity(e.target.value)}
              >
                <option value="Mild">Mild — functional, minimal distress</option>
                <option value="Moderate">Moderate — noticeably impairing daily routine</option>
                <option value="Severe">Severe — substantial distress, immediate attention</option>
                <option value="Critical">Critical — unstable vitals / red flag presentation</option>
              </select>
            </div>
          </div>
        </div>

        {/* ── Step 3: Patient History ── */}
        <div className="form-section">
          <div className="form-section-header">
            <div className="form-step-badge">03</div>
            <div>
              <div className="form-section-title">Patient History &amp; Notes</div>
              <div className="form-section-desc">Background context for refined analysis</div>
            </div>
          </div>

          <div className="form-field">
            <label className="field-label" htmlFor="notes-input">
              <FileText size={14} style={{ display: 'inline', marginRight: '6px', verticalAlign: '-2px' }} />
              Clinical Notes &amp; Background
            </label>
            <textarea
              id="notes-input"
              className="textarea-field"
              value={additionalNotes}
              onChange={(e) => setAdditionalNotes(e.target.value)}
              placeholder="Add relevant patient background: age, biological sex, comorbidities, recent travel, medication changes, or physical exam findings..."
            />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="error-banner">
            <AlertCircle size={18} />
            <div><strong>Notice:</strong> {error}</div>
          </div>
        )}

        {/* Submit Bar */}
        <div className="form-submit-row">
          <span className="submit-helper">
            Evidence-based knowledge matching with AI-assisted clinical reasoning.
          </span>
          <button
            type="submit"
            className="btn-diagnose"
            disabled={loading || selectedSymptoms.length === 0}
          >
            <Activity size={18} />
            <span>{loading ? 'Analyzing...' : 'Generate Differential Diagnosis'}</span>
          </button>
        </div>
      </form>
    </section>
  )
}
