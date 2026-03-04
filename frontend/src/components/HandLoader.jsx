import React from 'react'

export default function HandLoader({ fullPage = false }) {
  const inner = (
    <div className="hand">
      <div className="hand-finger" />
      <div className="hand-finger" />
      <div className="hand-finger" />
      <div className="hand-finger" />
      <div className="hand-palm" />
      <div className="hand-thumb" />
    </div>
  )

  if (!fullPage) return inner

  return (
    <div className="d-flex justify-content-center align-items-center vh-100">
      {inner}
    </div>
  )
}
