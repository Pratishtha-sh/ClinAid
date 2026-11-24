/**
 * ClinAid Diagnosis API client
 */

const API_BASE = '/api'
const DIRECT_BACKEND = 'http://localhost:8000'

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) })
    if (res.ok) return await res.json()
  } catch (err) {
    // try direct fallback
    try {
      const res = await fetch(`${DIRECT_BACKEND}/health`, { signal: AbortSignal.timeout(3000) })
      if (res.ok) return await res.json()
    } catch {
      return { status: 'offline', error: 'Backend unreachable' }
    }
  }
  return { status: 'offline' }
}

export async function submitDiagnosis({ symptoms, additional_info }) {
  const payload = {
    symptoms,
    additional_info: additional_info?.trim() || null,
  }

  let response
  try {
    response = await fetch(`${API_BASE}/diagnose`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch (e) {
    // Fallback directly to localhost:8000 if proxy isn't used
    response = await fetch(`${DIRECT_BACKEND}/diagnose`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `Diagnostic service error (Status ${response.status})`)
  }

  return await response.json()
}

export async function approveKG(items) {
  const payload = { items }

  let response
  try {
    response = await fetch(`${API_BASE}/approve-kg`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch (e) {
    response = await fetch(`${DIRECT_BACKEND}/approve-kg`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `Approval failed (Status ${response.status})`)
  }

  return await response.json()
}
