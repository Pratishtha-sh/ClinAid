import React, { useEffect, useState, useRef } from 'react'
import { Play, Pause, RotateCcw, ArrowRight, Activity, Info } from 'lucide-react'

const SVGNS = 'http://www.w3.org/2000/svg'

function el(tag, attrs) {
  const e = document.createElementNS(SVGNS, tag)
  for (const k in attrs) e.setAttribute(k, attrs[k])
  return e
}

export default function EvidenceAnimation({ inputSymptoms = [], candidates = [] }) {
  const [stepInfo, setStepInfo] = useState({
    current: 1,
    total: 1,
    label: 'Initializing evidence propagation...',
    isLast: false,
  })
  const [isPlaying, setIsPlaying] = useState(false)
  const timerRef = useRef(null)

  // Internal state tracking matching the user's animation logic
  const animState = useRef({
    steps: [],
    stepIndex: 0,
    scores: {},
    diseaseData: [],
    symptomData: [],
    missingData: [],
  })

  const runStep = () => {
    const { steps, stepIndex } = animState.current
    if (steps.length > 0 && stepIndex < steps.length) {
      steps[stepIndex].run()
      const isLast = stepIndex >= steps.length - 1
      setStepInfo({
        current: stepIndex + 1,
        total: steps.length,
        label: steps[stepIndex].label,
        isLast,
      })
      const nextBtn = document.getElementById('next-btn')
      if (nextBtn) {
        nextBtn.textContent = isLast ? 'Replay Animation' : 'Next Step →'
      }
      if (isLast) {
        setIsPlaying(false)
      }
    }
  }

  const updateBar = (id) => {
    const bar = document.getElementById('bar-' + id)
    if (!bar) return
    const scoreVal = animState.current.scores[id] || 0
    const w = Math.max(0, Math.min(160, scoreVal * 60))
    bar.setAttribute('width', w.toFixed(0))
  }

  const finalize = () => {
    const { diseaseData } = animState.current
    const ranked = [...diseaseData].sort((a, b) => b.final_score - a.final_score)
    ranked.forEach((d, i) => {
      const node = document.getElementById('node-' + d.id)
      const rankTxt = document.getElementById('rank-' + d.id)
      const bar = document.getElementById('bar-' + d.id)
      if (i < 3) {
        if (node) node.style.opacity = '1'
        if (rankTxt) rankTxt.textContent = '#' + (i + 1)
        if (bar) bar.setAttribute('fill', '#1D9E75')
      } else {
        if (node) node.style.opacity = '0.25'
        if (rankTxt) rankTxt.textContent = 'excluded'
      }
    })
  }

  const buildSteps = () => {
    const { symptomData, diseaseData, missingData } = animState.current
    const steps = []

    steps.push({
      label: 'Patient-reported symptoms verified against clinical database',
      run: () => {
        // Initial state
      },
    })

    symptomData.forEach((s) => {
      steps.push({
        label: `Matching "${s.name}" across candidate diseases`,
        run: () => {
          diseaseData.forEach((d) => {
            const match = (d.matched_symptoms || []).find(
              (m) => m.symptom.toLowerCase() === s.name.toLowerCase()
            )
            const line = document.getElementById('edge-' + s.id + '-' + d.id)
            if (match && line) {
              line.style.opacity = Math.max(0.2, match.weight)
              animState.current.scores[d.id] = (animState.current.scores[d.id] || 0) + (match.contribution || (match.weight * (match.specificity || 1)))
              updateBar(d.id)
            }
          })
        },
      })
    })

    if (missingData.length > 0) {
      steps.push({
        label: 'Penalizing diseases missing an expected symptom',
        run: () => {
          missingData.forEach((m) => {
            const ghostNode = document.getElementById(m.id)
            if (ghostNode) ghostNode.style.opacity = '1'

            diseaseData.forEach((d) => {
              const found = (d.missing_symptoms || []).find((x) => x.symptom === m.symptom)
              const line = document.getElementById('medge-' + m.id + '-' + d.id)
              if (found && line) {
                line.style.opacity = '0.5'
                const penalty = found.weight * (found.specificity || 1)
                animState.current.scores[d.id] = Math.max(0, (animState.current.scores[d.id] || 0) - penalty)
                const bar = document.getElementById('bar-' + d.id)
                if (bar) bar.setAttribute('fill', '#D85A30')
                updateBar(d.id)
              }
            })
          })
        },
      })
    }

    steps.push({
      label: 'Final differential ranking & score convergence',
      run: finalize,
    })

    animState.current.steps = steps
  }

  const buildAnimation = (inputSyms, candList) => {
    const edgeLayer = document.getElementById('edge-layer')
    const symLayer = document.getElementById('symptom-layer')
    const ghostLayer = document.getElementById('ghost-layer')
    const disLayer = document.getElementById('disease-layer')

    if (!edgeLayer || !symLayer || !ghostLayer || !disLayer) return

    ;[edgeLayer, symLayer, ghostLayer, disLayer].forEach((l) => (l.innerHTML = ''))

    const sx = 130, dx = 500
    const rowGap = 70, topPad = 55

    // Layout: symptom nodes (left column)
    const symptomData = inputSyms.map((name, i) => ({
      id: 'sym' + i,
      name,
      y: topPad + i * rowGap,
    }))

    // Layout: disease nodes (right column)
    const diseaseData = candList.map((c, i) => ({
      id: 'dis' + i,
      ...c,
      y: topPad + i * (rowGap + 15),
    }))

    // Unique missing symptoms across all candidates (top 5 by weight), placed below symptom column
    const missingMap = new Map()
    diseaseData.forEach((d) =>
      (d.missing_symptoms || []).forEach((m) => {
        if (!missingMap.has(m.symptom) || missingMap.get(m.symptom).weight < m.weight) {
          missingMap.set(m.symptom, m)
        }
      })
    )
    const missingData = [...missingMap.values()]
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 5)
      .map((m, i) => ({
        id: 'ghost' + i,
        ...m,
        y: topPad + symptomData.length * rowGap + 30 + i * 45,
      }))

    animState.current.symptomData = symptomData
    animState.current.diseaseData = diseaseData
    animState.current.missingData = missingData
    animState.current.scores = {}

    // Draw symptom nodes
    symptomData.forEach((s) => {
      const g = el('g', { class: 'node symptom-node' })
      g.appendChild(el('rect', { x: sx - 75, y: s.y - 18, width: 150, height: 36, rx: 6 }))
      const t = el('text', { class: 'title', x: sx, y: s.y + 4, 'text-anchor': 'middle' })
      t.textContent = s.name.length > 18 ? s.name.slice(0, 17) + '…' : s.name
      const titleEl = el('title', {})
      titleEl.textContent = s.name
      g.appendChild(titleEl)
      g.appendChild(t)
      symLayer.appendChild(g)
    })

    // Draw ghost (missing) nodes, hidden until their step
    missingData.forEach((m) => {
      const g = el('g', {
        class: 'node ghost',
        id: m.id,
        style: 'opacity:0; transition:opacity .4s',
      })
      g.appendChild(el('rect', { x: sx - 75, y: m.y - 16, width: 150, height: 32, rx: 6 }))
      const t = el('text', { class: 'title', x: sx, y: m.y + 4, 'text-anchor': 'middle' })
      const textVal = m.symptom + ' (absent)'
      t.textContent = textVal.length > 20 ? textVal.slice(0, 19) + '…' : textVal
      const titleEl = el('title', {})
      titleEl.textContent = `${m.symptom} (Missing expected symptom)`
      g.appendChild(titleEl)
      g.appendChild(t)
      ghostLayer.appendChild(g)
    })

    // Draw disease nodes + score bars
    diseaseData.forEach((d) => {
      animState.current.scores[d.id] = 0
      const g = el('g', {
        class: 'node disease-node',
        id: 'node-' + d.id,
        style: 'transition:opacity .5s',
      })
      g.appendChild(el('rect', { x: dx - 80, y: d.y - 18, width: 160, height: 36, rx: 6 }))
      const t = el('text', { class: 'title', x: dx, y: d.y - 1, 'text-anchor': 'middle' })
      t.textContent = d.disease.length > 20 ? d.disease.slice(0, 19) + '…' : d.disease
      const titleEl = el('title', {})
      titleEl.textContent = d.disease
      g.appendChild(titleEl)
      g.appendChild(t)
      g.appendChild(
        el('rect', {
          class: 'bar-bg',
          x: dx - 80,
          y: d.y + 22,
          width: 160,
          height: 6,
          rx: 3,
        })
      )
      g.appendChild(
        el('rect', {
          class: 'bar',
          id: 'bar-' + d.id,
          x: dx - 80,
          y: d.y + 22,
          width: 0,
          height: 6,
          rx: 3,
        })
      )
      const rankT = el('text', {
        class: 'rank',
        id: 'rank-' + d.id,
        x: dx + 90,
        y: d.y - 1,
        'text-anchor': 'start',
      })
      g.appendChild(rankT)
      disLayer.appendChild(g)
    })

    // Draw all matched-symptom edges (hidden initially)
    symptomData.forEach((s) => {
      diseaseData.forEach((d) => {
        const match = (d.matched_symptoms || []).find(
          (m) => m.symptom.toLowerCase() === s.name.toLowerCase()
        )
        if (!match) return
        edgeLayer.appendChild(
          el('line', {
            class: 'edge',
            id: 'edge-' + s.id + '-' + d.id,
            x1: sx + 75,
            y1: s.y,
            x2: dx - 80,
            y2: d.y,
            stroke: '#7F77DD',
            'stroke-width': Math.max(1.5, (match.weight || 0.5) * 5),
            opacity: 0,
            'marker-end': 'url(#arrow)',
          })
        )
      })
    })

    // Draw missing-symptom edges (hidden initially)
    missingData.forEach((m) => {
      diseaseData.forEach((d) => {
        const found = (d.missing_symptoms || []).find((x) => x.symptom === m.symptom)
        if (!found) return
        edgeLayer.appendChild(
          el('line', {
            class: 'edge',
            id: 'medge-' + m.id + '-' + d.id,
            x1: sx + 75,
            y1: m.y,
            x2: dx - 80,
            y2: d.y,
            stroke: '#D85A30',
            'stroke-width': Math.max(1.5, (found.weight || 0.5) * 4),
            'stroke-dasharray': '3 3',
            opacity: 0,
          })
        )
      })
    })

    // Resize the SVG to fit whatever content was built
    const allY = [
      ...symptomData.map((s) => s.y),
      ...missingData.map((m) => m.y),
      ...diseaseData.map((d) => d.y),
    ]
    const maxY = (allY.length > 0 ? Math.max(...allY) : 260) + 60

    const svgCanvas = document.getElementById('svg-canvas')
    if (svgCanvas) {
      svgCanvas.setAttribute('viewBox', `0 0 680 ${Math.max(280, maxY)}`)
    }

    buildSteps()
    animState.current.stepIndex = 0

    const nextBtn = document.getElementById('next-btn')
    if (nextBtn) {
      nextBtn.disabled = false
      nextBtn.textContent = 'Next Step →'
    }

    runStep()
  }

  // Handle next step
  const handleNext = () => {
    const { steps, stepIndex } = animState.current
    if (stepIndex >= steps.length - 1) {
      buildAnimation(inputSymptoms, candidates)
    } else {
      animState.current.stepIndex++
      runStep()
    }
  }

  // Handle replay / reset
  const handleReset = () => {
    setIsPlaying(false)
    if (timerRef.current) clearInterval(timerRef.current)
    buildAnimation(inputSymptoms, candidates)
  }

  // Toggle Auto Play
  const togglePlay = () => {
    setIsPlaying((prev) => !prev)
  }

  // Auto-play timer effect
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        const { steps, stepIndex } = animState.current
        if (stepIndex >= steps.length - 1) {
          setIsPlaying(false)
        } else {
          animState.current.stepIndex++
          runStep()
        }
      }, 1300)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isPlaying])

  // Build animation when inputs change or component mounts
  useEffect(() => {
    if (inputSymptoms.length > 0 || candidates.length > 0) {
      // Small timeout to guarantee DOM nodes are mounted
      const t = setTimeout(() => {
        buildAnimation(inputSymptoms, candidates)
      }, 50)
      return () => clearTimeout(t)
    }
  }, [inputSymptoms, candidates])

  return (
    <div className="evidence-animation-card">
      <div className="animation-header">
        <div>
          <div className="anim-badge">
            <Activity size={15} color="#0b2545" />
            <span>Interactive Evidence Propagation Pipeline</span>
          </div>
          <h3 className="anim-title">Diagnostic Scoring & Evidence Visualization</h3>
          <p className="anim-desc">
            Visualizing association weights from patient-reported symptoms to candidate conditions and absent expected symptom penalties.
          </p>
        </div>

        {/* Legend */}
        <div className="anim-legend">
          <div className="legend-item">
            <span className="legend-dot" style={{ background: '#0b2545' }} />
            <span>Reported Symptom</span>
          </div>
          <div className="legend-item">
            <span className="legend-line" style={{ background: '#7F77DD' }} />
            <span>Positive Weight</span>
          </div>
          <div className="legend-item">
            <span className="legend-line dashed" style={{ borderColor: '#D85A30' }} />
            <span>Absent Penalty</span>
          </div>
          <div className="legend-item">
            <span className="legend-dot" style={{ background: '#1D9E75' }} />
            <span>Final Candidate (#)</span>
          </div>
        </div>
      </div>

      {/* Control Bar */}
      <div className="anim-controls-bar">
        <div className="step-indicator">
          <span className="step-pill">
            Step {stepInfo.current} / {stepInfo.total}
          </span>
          <span id="step-label" className="step-label">
            {stepInfo.label}
          </span>
        </div>

        <div className="anim-buttons">
          <button
            id="next-btn"
            type="button"
            className="anim-btn primary"
            onClick={handleNext}
          >
            {stepInfo.isLast ? 'Replay Animation' : 'Next Step →'}
          </button>

          <button
            type="button"
            className={`anim-btn secondary ${isPlaying ? 'playing' : ''}`}
            onClick={togglePlay}
          >
            {isPlaying ? (
              <>
                <Pause size={14} />
                <span>Pause</span>
              </>
            ) : (
              <>
                <Play size={14} />
                <span>Auto Play</span>
              </>
            )}
          </button>

          <button
            type="button"
            className="anim-btn secondary"
            onClick={handleReset}
            title="Reset to step 1"
          >
            <RotateCcw size={14} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* SVG Canvas Container */}
      <div className="svg-wrapper">
        <svg
          id="svg-canvas"
          viewBox="0 0 680 340"
          className="evidence-svg"
        >
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="6"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#7F77DD" />
            </marker>
          </defs>
          <g id="edge-layer" />
          <g id="symptom-layer" />
          <g id="ghost-layer" />
          <g id="disease-layer" />
        </svg>
      </div>
    </div>
  )
}
