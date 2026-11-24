import React, { useState, useEffect, Component } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import DiagnosisForm from './components/DiagnosisForm'
import DiagnosisResults from './components/DiagnosisResults'
import { checkHealth, submitDiagnosis } from './services/api'
import { Activity, AlertTriangle } from 'lucide-react'

class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('Diagnosis render error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          margin: '2rem 0',
          padding: '1.75rem',
          background: '#fff1f2',
          border: '1px solid #fecdd3',
          borderRadius: '16px',
          color: '#9f1239',
          boxShadow: '0 4px 20px rgba(225, 29, 72, 0.08)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <AlertTriangle size={20} color="#e11d48" />
            <h4 style={{ fontWeight: 700, fontSize: '1.05rem', margin: 0 }}>Rendering Error</h4>
          </div>
          <p style={{ fontSize: '0.9rem', color: '#881337', marginBottom: '12px' }}>
            {this.state.error?.message || 'An error occurred while displaying diagnosis results.'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              padding: '8px 18px',
              background: '#e11d48',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            Reset Display
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Initial health check
    checkHealth().then((data) => setHealth(data))
    const interval = setInterval(() => {
      checkHealth().then((data) => setHealth(data))
    }, 15000)
    return () => clearInterval(interval)
  }, [])

  const handleDiagnose = async (formData) => {
    setLoading(true)
    setError(null)

    try {
      const data = await submitDiagnosis(formData)
      setResult({ ...data, input_symptoms: formData.symptoms })

      // Scroll to results area smoothly
      setTimeout(() => {
        const el = document.getElementById('results-area')
        if (el) el.scrollIntoView({ behavior: 'smooth' })
      }, 150)
    } catch (err) {
      console.error('Diagnosis request failed:', err)
      setError(
        err.message ||
        'Failed to connect to the Diagnosis API. Please ensure the backend is running at http://localhost:8000.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="clinaid-app">
      <Navbar health={health} />

      <main>
        {/* Full Laptop Screen Hero */}
        <Hero />

        {/* Interactive Diagnosis Form */}
        <DiagnosisForm
          onSubmit={handleDiagnose}
          loading={loading}
          error={error}
        />

        {/* Diagnosis Results */}
        <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 2rem' }}>
          <ErrorBoundary>
            <DiagnosisResults result={result} loading={loading} externalData={result?.external_data} />
          </ErrorBoundary>
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        marginTop: '0',
        borderTop: '1px solid rgba(255, 255, 255, 0.07)',
        padding: '3rem 2.5rem',
        background: '#06101e',
        textAlign: 'center',
        color: '#64748b',
        fontSize: '0.85rem'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '0.9rem', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '9px', color: '#ffffff', fontWeight: 700, fontFamily: "'Outfit', sans-serif", fontSize: '1rem' }}>
            <Activity size={18} color="#10b981" />
            <span>ClinAid</span>
          </div>
          <p style={{ color: '#374151', fontSize: '0.76rem', marginTop: '0.25rem' }}>
            © {new Date().getFullYear()} ClinAid. For clinical decision support and evaluation only.
          </p>
        </div>
      </footer>
    </div>
  )
}
