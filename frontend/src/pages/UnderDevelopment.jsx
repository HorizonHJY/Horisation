import React from 'react'
import { useNavigate } from 'react-router-dom'

export default function UnderDevelopment() {
  const navigate = useNavigate()
  return (
    <div className="under-dev-page">
      <div className="under-dev-icon">
        <i className="fas fa-hard-hat" />
      </div>
      <h2 className="fw-bold mb-2">Under Development</h2>
      <p className="text-muted mb-4" style={{ maxWidth: 400 }}>
        This section is currently being built. Check back soon!
      </p>
      <button className="btn btn-primary" onClick={() => navigate('/home')}>
        <i className="fas fa-home me-2" />Back to Home
      </button>
    </div>
  )
}
