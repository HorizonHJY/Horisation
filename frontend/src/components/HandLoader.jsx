import React from 'react'

export default function HandLoader({ fullPage = false }) {
  const inner = (
    <div className="newtons-cradle">
      <div className="newtons-cradle__dot" />
      <div className="newtons-cradle__dot" />
      <div className="newtons-cradle__dot" />
      <div className="newtons-cradle__dot" />
    </div>
  )

  if (!fullPage) return inner

  return (
    <div className="d-flex justify-content-center align-items-center vh-100">
      {inner}
    </div>
  )
}
