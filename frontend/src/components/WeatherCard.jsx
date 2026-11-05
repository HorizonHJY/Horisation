import React, { useState, useEffect, useCallback, lazy, Suspense } from 'react'

// Plotly is ~130 kB, so the 30-day chart is still lazy — nothing renders
// unless the user opens it.
const WeatherChart = lazy(() => import('./WeatherChart'))

/* ── Animated SVG scenes ─────────────────────────────────────────
   Hand-drawn with plain SVG + CSS keyframes (defined in index.css
   under the `.wc-` namespace). No new deps. `prefers-reduced-motion`
   is honoured at the CSS layer, so every scene falls to a still frame
   automatically. */

function SceneSun() {
  // Sun with slowly rotating rays and a soft breathing halo.
  return (
    <svg viewBox="0 0 240 160" className="wc-svg" role="img" aria-label="Sunny">
      <defs>
        <radialGradient id="wc-sun-halo" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="#ffd98a" stopOpacity="0.55" />
          <stop offset="60%"  stopColor="#ffd98a" stopOpacity="0.15" />
          <stop offset="100%" stopColor="#ffd98a" stopOpacity="0"    />
        </radialGradient>
      </defs>
      <circle cx="120" cy="80" r="62" fill="url(#wc-sun-halo)" className="wc-halo" />
      <g className="wc-sun-rays" style={{ transformOrigin: '120px 80px' }}>
        {Array.from({ length: 12 }).map((_, i) => (
          <rect key={i}
                x="118.5" y="12" width="3" height="18" rx="1.5"
                fill="#f5b942"
                transform={`rotate(${i * 30} 120 80)`} />
        ))}
      </g>
      <circle cx="120" cy="80" r="28" fill="#fcc24b" />
      <circle cx="120" cy="80" r="28" fill="none" stroke="#e39a1f" strokeOpacity="0.25" strokeWidth="2" />
    </svg>
  )
}

function Cloud({ x, y, scale = 1, opacity = 1, className = '' }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${scale})`} opacity={opacity} className={className}>
      <path d="M0 20
               a16 16 0 0 1 16 -16
               a18 18 0 0 1 34 -4
               a14 14 0 0 1 24 10
               a12 12 0 0 1 -6 22
               H6
               a14 14 0 0 1 -6 -12z"
            fill="#e9eef5" stroke="#c9d3e0" strokeWidth="1.2" />
    </g>
  )
}

function SceneCloudSun() {
  // Sun peeks out behind a drifting cloud.
  return (
    <svg viewBox="0 0 240 160" className="wc-svg" role="img" aria-label="Partly cloudy">
      <defs>
        <radialGradient id="wc-sun-halo-2" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="#ffd98a" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#ffd98a" stopOpacity="0"   />
        </radialGradient>
      </defs>
      <circle cx="95" cy="68" r="52" fill="url(#wc-sun-halo-2)" className="wc-halo" />
      <g className="wc-sun-rays" style={{ transformOrigin: '95px 68px' }}>
        {Array.from({ length: 12 }).map((_, i) => (
          <rect key={i}
                x="93.5" y="18" width="3" height="14" rx="1.5"
                fill="#f5b942"
                transform={`rotate(${i * 30} 95 68)`} />
        ))}
      </g>
      <circle cx="95" cy="68" r="22" fill="#fcc24b" />
      <g className="wc-drift-slow"><Cloud x={110} y={72} scale={1.25} /></g>
      <g className="wc-drift-fast"><Cloud x={30}  y={102} scale={0.75} opacity={0.75} /></g>
    </svg>
  )
}

function SceneCloud() {
  // Two layers drifting at different speeds.
  return (
    <svg viewBox="0 0 240 160" className="wc-svg" role="img" aria-label="Cloudy">
      <g className="wc-drift-slow"><Cloud x={60}  y={46} scale={1.35} /></g>
      <g className="wc-drift-fast"><Cloud x={20}  y={90} scale={0.95} opacity={0.85} /></g>
      <g className="wc-drift-mid" ><Cloud x={130} y={100} scale={1.1}  opacity={0.9}  /></g>
    </svg>
  )
}

function Raindrops({ heavy = false }) {
  const drops = heavy ? 16 : 10
  return (
    <g className="wc-rain">
      {Array.from({ length: drops }).map((_, i) => {
        const x = 40 + ((i * 17) % 160)
        const delay = ((i * 137) % 1000) / 1000
        return (
          <line key={i}
                x1={x} y1={70} x2={x - 4} y2={82}
                stroke="#6b9cdb" strokeWidth="2" strokeLinecap="round"
                className="wc-raindrop"
                style={{ animationDelay: `${delay}s` }} />
        )
      })}
    </g>
  )
}

function SceneRain({ heavy = false }) {
  return (
    <svg viewBox="0 0 240 160" className="wc-svg" role="img" aria-label={heavy ? 'Heavy rain' : 'Rain'}>
      <g className="wc-drift-slow"><Cloud x={70} y={34} scale={1.4} /></g>
      <g className="wc-drift-mid" ><Cloud x={20} y={56} scale={0.95} opacity={0.85} /></g>
      <Raindrops heavy={heavy} />
    </svg>
  )
}

function SceneSnow() {
  const flakes = 14
  return (
    <svg viewBox="0 0 240 160" className="wc-svg" role="img" aria-label="Snow">
      <g className="wc-drift-slow"><Cloud x={70} y={34} scale={1.4} /></g>
      <g className="wc-drift-mid" ><Cloud x={20} y={56} scale={0.95} opacity={0.85} /></g>
      <g className="wc-snow">
        {Array.from({ length: flakes }).map((_, i) => {
          const x = 30 + ((i * 15) % 180)
          const delay = ((i * 211) % 1000) / 1000
          const dur = 3 + ((i * 97) % 20) / 10
          return (
            <circle key={i} cx={x} cy={72} r="2.2" fill="#eaf2fb"
                    className="wc-snowflake"
                    style={{ animationDelay: `${delay}s`, animationDuration: `${dur}s` }} />
          )
        })}
      </g>
    </svg>
  )
}

function SceneBolt() {
  return (
    <svg viewBox="0 0 240 160" className="wc-svg" role="img" aria-label="Thunderstorm">
      <g className="wc-drift-slow"><Cloud x={60} y={30} scale={1.45} /></g>
      <g className="wc-drift-mid" ><Cloud x={15} y={54} scale={0.95} opacity={0.85} /></g>
      <Raindrops />
      <path d="M128 76 L112 108 L126 108 L118 134 L142 98 L128 98 Z"
            fill="#f5c842" stroke="#c68f10" strokeWidth="1.2"
            className="wc-bolt" />
    </svg>
  )
}

function SceneFog() {
  return (
    <svg viewBox="0 0 240 160" className="wc-svg" role="img" aria-label="Fog">
      <g className="wc-drift-slow"><Cloud x={60} y={28} scale={1.2} opacity={0.6} /></g>
      <g className="wc-fog">
        {[60, 82, 104, 126].map((y, i) => (
          <rect key={i}
                x="-40" y={y} width="320" height="10" rx="5"
                fill="#cfd8e3" opacity={0.55 - i * 0.08}
                className={`wc-fog-band wc-fog-band--${i}`} />
        ))}
      </g>
    </svg>
  )
}

const SCENES = {
  'fa-sun':                 SceneSun,
  'fa-cloud-sun':           SceneCloudSun,
  'fa-cloud':               SceneCloud,
  'fa-cloud-rain':          (p) => <SceneRain {...p} heavy={false} />,
  'fa-cloud-showers-heavy': (p) => <SceneRain {...p} heavy={true} />,
  'fa-snowflake':           SceneSnow,
  'fa-smog':                SceneFog,
  'fa-bolt':                SceneBolt,
}

function WeatherScene({ icon }) {
  const Scene = SCENES[icon] || SceneCloudSun
  return <Scene />
}

function todayLabel() {
  return new Date().toLocaleDateString('en-US',
    { weekday: 'short', month: 'short', day: 'numeric', timeZone: 'America/Chicago' })
}

function fallbackGreeting(name) {
  const chicagoHour = Number(new Date().toLocaleString('en-US',
    { timeZone: 'America/Chicago', hour: 'numeric', hour12: false }))
  const greet = chicagoHour < 12 ? 'Good morning'
               : chicagoHour < 18 ? 'Good afternoon'
               : 'Good evening'
  return `${greet}, ${name}`
}

export default function WeatherCard({ name }) {
  const [weather, setWeather] = useState(null)
  const [failed, setFailed]   = useState(false)
  const [showChart, setShowChart] = useState(false)

  const fetchWeather = useCallback(async () => {
    try {
      const resp = await fetch('/api/weather', { credentials: 'include' })
      const data = await resp.json()
      if (data.ok) { setWeather(data); setFailed(false) }
      else         { setFailed(true) }
    } catch {
      setFailed(true)
    }
  }, [])

  useEffect(() => {
    fetchWeather()
    const interval = setInterval(fetchWeather, 10 * 60 * 1000)
    return () => clearInterval(interval)
  }, [fetchWeather])

  const hasData  = !!weather && weather.temp_f != null
  const greeting = hasData ? weather.greeting.replace('{name}', name)
                           : fallbackGreeting(name)
  const date     = todayLabel()
  const icon     = hasData ? weather.icon : 'fa-cloud-sun'

  return (
    <div className="hero-block hero-block--weather wc-card">
      <div className="wc-grid">
        <div className="wc-illus" aria-hidden={failed || !hasData ? 'true' : undefined}>
          <WeatherScene icon={icon} />
        </div>

        <div className="wc-info">
          <h2 className="hero-title wc-greeting">{greeting}</h2>

          <div className="wc-meta">
            <span className="wc-date">{date}</span>
            {hasData && (
              <>
                <span className="wc-dot" aria-hidden="true">·</span>
                <span className="wc-city">{weather.city}</span>
              </>
            )}
          </div>

          {hasData ? (
            <>
              <div className="wc-temp-row">
                <span className="wc-temp-big">{weather.temp_f}°<span className="wc-unit">F</span></span>
                <span className="wc-label">{weather.label}</span>
              </div>
              <div className="wc-feels">
                Feels like {weather.feels_like_f}°F / {weather.feels_like_c}°C
              </div>
              {weather.tomorrow && (
                <div className="wc-tomorrow">
                  <span className="wc-tomorrow-pill">Tomorrow</span>
                  <span className="wc-tomorrow-label">{weather.tomorrow.label}</span>
                  <span className="wc-tomorrow-temp">
                    <span className="wc-hi">H</span> {weather.tomorrow.high_f}°
                    {weather.tomorrow.low_f != null && (
                      <>  <span className="wc-lo">L</span> {weather.tomorrow.low_f}°</>
                    )}
                  </span>
                </div>
              )}
            </>
          ) : (
            <div className="wc-placeholder">
              {failed ? 'Weather unavailable right now.' : 'Loading weather…'}
            </div>
          )}
        </div>
      </div>

      <button type="button" className="wg-trend-toggle wc-trend-toggle"
              onClick={() => setShowChart(v => !v)}
              aria-expanded={showChart}>
        {showChart ? 'Hide 30-day trend' : 'See 30-day trend'}
        <span className="wg-trend-caret" aria-hidden="true">{showChart ? '⌃' : '⌄'}</span>
      </button>
      {showChart && (
        <Suspense fallback={<div className="small text-muted py-3">Loading chart…</div>}>
          <WeatherChart />
        </Suspense>
      )}
    </div>
  )
}
