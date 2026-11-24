import React from 'react'
import { ArrowDown, Stethoscope } from 'lucide-react'

export default function Hero() {
  const scrollToForm = () => {
    const el = document.getElementById('diagnosis-section')
    if (el) el.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <section className="hero-section">
      <div className="hero-container">
        {/* Left Column */}
        <div className="hero-left">
          <p className="hero-eyebrow">Evidence-Based Clinical Decision Support</p>

          <h1 className="hero-title">
            Differential Diagnosis<br />
            <span className="highlight">For Clinicians</span>
          </h1>

          <p className="hero-description">
            ClinAid delivers structured, transparent differential diagnosis by matching
            patient symptoms against a curated medical knowledge graph — providing ranked
            disease candidates with clear clinical rationale.
          </p>

          <div className="hero-stats-row">
            <div className="stat-item">
              <span className="stat-val">773+</span>
              <span className="stat-label">Diseases in Knowledge Base</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-val">377</span>
              <span className="stat-label">Clinical Symptom Entities</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-val">5,389</span>
              <span className="stat-label">Validated Associations</span>
            </div>
          </div>

          <div className="hero-cta-group">
            <button className="btn-primary" onClick={scrollToForm}>
              <span>Begin Clinical Assessment</span>
              <ArrowDown size={17} />
            </button>
            <a href="#diagnosis-section" className="btn-outline">
              <Stethoscope size={16} />
              <span>Explore the Engine</span>
            </a>
          </div>
        </div>

        {/* Right Column: Collage */}
        <div className="hero-right">
          <div className="collage-perspective-wrapper">
            <div className="collage-bg-card card-top-right">
              <img
                src="/images/accompaniment-abortion-process.jpg"
                alt="Clinical consultation and patient care"
                loading="eager"
              />
              <div className="card-overlay">
                <p>Compassionate Clinical Care</p>
              </div>
            </div>

            <div className="collage-bg-card card-middle-right">
              <img
                src="/images/filling-medical-record.jpg"
                alt="Doctor filling clinical medical record"
                loading="eager"
              />
              <div className="card-overlay">
                <p>EHR &amp; Differential Triaging</p>
              </div>
            </div>

            <div className="collage-bg-card card-bottom-center">
              <img
                src="/images/patient-telling-doctor-about-symptoms-disease.jpg"
                alt="Patient discussing clinical symptoms with physician"
                loading="eager"
              />
              <div className="card-overlay">
                <p>Symptom Assessment &amp; Dialogue</p>
              </div>
            </div>

            <div className="doctor-hero-frame">
              <img
                src="/images/doctor_hero.jpg"
                alt="ClinAid Lead Physician"
                loading="eager"
                onError={(e) => {
                  e.target.src = '/images/vitaly-gariev-JJEOuvnY1Tw-unsplash.jpg'
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
