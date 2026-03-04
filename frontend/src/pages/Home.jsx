import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../App'

export default function Home() {
  const { user } = useAuth()
  const navigate = useNavigate()

  return (
    <div
      className="rounded-3 text-white p-4 p-md-5"
      style={{ background: 'linear-gradient(135deg, #1e2a3a 0%, #3a7bd5 100%)' }}
    >
      <h2 className="fw-bold mb-1">
        Hello, {user?.display_name} 👋
      </h2>
      <p className="mb-3 opacity-75">Welcome to your personal platform. What are you working on today?</p>
      <button
        className="btn btn-light btn-sm fw-semibold"
        onClick={() => navigate('/hormemo')}
      >
        <i className="fas fa-clipboard-list me-2" />Open Hormemo
      </button>
    </div>
  )
}
