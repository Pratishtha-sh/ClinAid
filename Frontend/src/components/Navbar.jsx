import React from 'react'

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="nav-container">
        {/* Brand — logo + text */}
        <div className="nav-brand">
          <img
            src="/images/logo.png?v=2"
            alt="ClinAid Logo"
            className="brand-logo"
          />
          <span className="brand-name">ClinAid</span>
        </div>

        {/* Navigation Links */}
        <nav className="nav-links">
          <a href="#diagnosis-section" className="nav-link">Differential Engine</a>
          <a href="#results-area" className="nav-link">Clinical Results</a>
          <a href="#diagnosis-section" className="nav-link">Documentation</a>
        </nav>

        {/* CTA Button */}
        <a href="#diagnosis-section" className="nav-cta">
          Start Assessment
        </a>
      </div>
    </header>
  )
}
