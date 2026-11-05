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

// Star positions used for the clear-night scene. Each twinkles on its own
// delay so the sparkle feels organic, not metronomic.
const NIGHT_STARS = [
  { cx:  60, cy:  34, r: 1.6, delay: 0.0 },
  { cx: 190, cy:  28, r: 2.0, delay: 0.7 },
  { cx: 210, cy:  96, r: 1.4, delay: 1.3 },
  { cx:  38, cy: 110, r: 1.8, delay: 2.1 },
  { cx: 165, cy: 132, r: 1.5, delay: 2.8 },
]

function Star({ cx, cy, r, delay }) {
  return (
    <circle cx={cx} cy={cy} r={r} fill="#fdf6c7"
            className="wc-star" style={{ animationDelay: `${delay}s` }} />
  )
}

function SceneMoon() {
  // Crescent moon with soft halo, plus a handful of twinkling stars.
  return (
    <svg viewBox="0 0 240 160" className="wc-svg wc-night" role="img" aria-label="Clear night">
      <defs>
        <radialGradient id="wc-moon-halo" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="#dce4ff" stopOpacity="0.55" />
          <stop offset="60%"  stopColor="#dce4ff" stopOpacity="0.15" />
          <stop offset="100%" stopColor="#dce4ff" stopOpacity="0"    />
        </radialGradient>
        {/* Carve the crescent by masking out an offset disk. */}
        <mask id="wc-moon-mask">
          <rect x="0" y="0" width="240" height="160" fill="#000" />
          <circle cx="120" cy="80" r="32" fill="#fff" />
          <circle cx="134" cy="72" r="28" fill="#000" />
        </mask>
      </defs>
      <circle cx="120" cy="80" r="56" fill="url(#wc-moon-halo)" className="wc-halo" />
      <rect x="0" y="0" width="240" height="160" fill="#f6efc7" mask="url(#wc-moon-mask)" />
      {NIGHT_STARS.map((s, i) => <Star key={i} {...s} />)}
    </svg>
  )
}

function Cloud({ x, y, scale = 1, opacity = 1, className = '', night = false }) {
  const fill   = night ? '#3a4660' : '#e9eef5'
  const stroke = night ? '#243049' : '#c9d3e0'
  return (
    <g transform={`translate(${x} ${y}) scale(${scale})`} opacity={opacity} className={className}>
      <path d="M0 20
               a16 16 0 0 1 16 -16
               a18 18 0 0 1 34 -4
               a14 14 0 0 1 24 10
               a12 12 0 0 1 -6 22
               H6
               a14 14 0 0 1 -6 -12z"
            fill={fill} stroke={stroke} strokeWidth="1.2" />
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

function SceneCloudMoon() {
  // Night version of partly-cloudy: moon tucked behind a drifting cloud,
  // plus two small stars so the sky is not empty.
  return (
    <svg viewBox="0 0 240 160" className="wc-svg wc-night" role="img" aria-label="Partly cloudy night">
      <defs>
        <radialGradient id="wc-moon-halo-2" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="#dce4ff" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#dce4ff" stopOpacity="0"    />
        </radialGradient>
        <mask id="wc-moon-mask-2">
          <rect x="0" y="0" width="240" height="160" fill="#000" />
          <circle cx="95" cy="68" r="22" fill="#fff" />
          <circle cx="105" cy="62" r="19" fill="#000" />
        </mask>
      </defs>
      <circle cx="95" cy="68" r="44" fill="url(#wc-moon-halo-2)" className="wc-halo" />
      <rect x="0" y="0" width="240" height="160" fill="#f6efc7" mask="url(#wc-moon-mask-2)" />
      <Star cx={180} cy={36} r={1.6} delay={0.4} />
      <Star cx={200} cy={100} r={1.4} delay={1.6} />
      <g className="wc-drift-slow"><Cloud x={110} y={72} scale={1.25} night /></g>
      <g className="wc-drift-fast"><Cloud x={30}  y={102} scale={0.75} opacity={0.85} night /></g>
    </svg>
  )
}

function SceneCloud({ night = false }) {
  // Two layers drifting at different speeds.
  return (
    <svg viewBox="0 0 240 160" className={`wc-svg${night ? ' wc-night' : ''}`} role="img" aria-label="Cloudy">
      <g className="wc-drift-slow"><Cloud x={60}  y={46} scale={1.35} night={night} /></g>
      <g className="wc-drift-fast"><Cloud x={20}  y={90} scale={0.95} opacity={0.85} night={night} /></g>
      <g className="wc-drift-mid" ><Cloud x={130} y={100} scale={1.1}  opacity={0.9}  night={night} /></g>
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

function SceneRain({ heavy = false, night = false }) {
  return (
    <svg viewBox="0 0 240 160" className={`wc-svg${night ? ' wc-night' : ''}`} role="img" aria-label={heavy ? 'Heavy rain' : 'Rain'}>
      <g className="wc-drift-slow"><Cloud x={70} y={34} scale={1.4} night={night} /></g>
      <g className="wc-drift-mid" ><Cloud x={20} y={56} scale={0.95} opacity={0.85} night={night} /></g>
      <Raindrops heavy={heavy} />
    </svg>
  )
}

function SceneSnow({ night = false }) {
  const flakes = 14
  return (
    <svg viewBox="0 0 240 160" className={`wc-svg${night ? ' wc-night' : ''}`} role="img" aria-label="Snow">
      <g className="wc-drift-slow"><Cloud x={70} y={34} scale={1.4} night={night} /></g>
      <g className="wc-drift-mid" ><Cloud x={20} y={56} scale={0.95} opacity={0.85} night={night} /></g>
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

function SceneBolt({ night = false }) {
  return (
    <svg viewBox="0 0 240 160" className={`wc-svg${night ? ' wc-night' : ''}`} role="img" aria-label="Thunderstorm">
      <g className="wc-drift-slow"><Cloud x={60} y={30} scale={1.45} night={night} /></g>
      <g className="wc-drift-mid" ><Cloud x={15} y={54} scale={0.95} opacity={0.85} night={night} /></g>
      <Raindrops />
      <path d="M128 76 L112 108 L126 108 L118 134 L142 98 L128 98 Z"
            fill="#f5c842" stroke="#c68f10" strokeWidth="1.2"
            className="wc-bolt" />
    </svg>
  )
}

function SceneFog({ night = false }) {
  return (
    <svg viewBox="0 0 240 160" className={`wc-svg${night ? ' wc-night' : ''}`} role="img" aria-label="Fog">
      <g className="wc-drift-slow"><Cloud x={60} y={28} scale={1.2} opacity={0.6} night={night} /></g>
      <g className="wc-fog">
        {[60, 82, 104, 126].map((y, i) => (
          <rect key={i}
                x="-40" y={y} width="320" height="10" rx="5"
                fill={night ? '#5a6478' : '#cfd8e3'} opacity={0.55 - i * 0.08}
                className={`wc-fog-band wc-fog-band--${i}`} />
        ))}
      </g>
    </svg>
  )
}

function WeatherScene({ icon, night }) {
  // Pick a scene; swap sunny/partly-cloudy for their lunar siblings at night.
  if (night) {
    switch (icon) {
      case 'fa-sun':                 return <SceneMoon />
      case 'fa-cloud-sun':           return <SceneCloudMoon />
      case 'fa-cloud':               return <SceneCloud night />
      case 'fa-cloud-rain':          return <SceneRain  night heavy={false} />
      case 'fa-cloud-showers-heavy': return <SceneRain  night heavy={true}  />
      case 'fa-snowflake':           return <SceneSnow  night />
      case 'fa-smog':                return <SceneFog   night />
      case 'fa-bolt':                return <SceneBolt  night />
      default:                       return <SceneCloudMoon />
    }
  }
  switch (icon) {
    case 'fa-sun':                 return <SceneSun />
    case 'fa-cloud-sun':           return <SceneCloudSun />
    case 'fa-cloud':               return <SceneCloud />
    case 'fa-cloud-rain':          return <SceneRain heavy={false} />
    case 'fa-cloud-showers-heavy': return <SceneRain heavy={true}  />
    case 'fa-snowflake':           return <SceneSnow />
    case 'fa-smog':                return <SceneFog />
    case 'fa-bolt':                return <SceneBolt />
    default:                       return <SceneCloudSun />
  }
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

// Night detection — prefer the backend's sunrise/sunset (ISO wall-clock in
// America/Chicago, no offset suffix); fall back to hour-of-day when the
// backend hasn't shipped those fields yet.
function isNightNow(sunrise, sunset) {
  // Current wall-clock minutes-since-midnight in Chicago.
  const nowStr = new Date().toLocaleString('en-US', {
    timeZone: 'America/Chicago', hour12: false,
    hour: '2-digit', minute: '2-digit',
  })
  const [h, m] = nowStr.split(':').map(Number)
  const nowMin = h * 60 + m

  const parse = (iso) => {
    // Open-Meteo returns e.g. "2026-11-04T06:40" (local, no tz).
    const t = (iso || '').split('T')[1]
    if (!t) return null
    const [hh, mm] = t.split(':').map(Number)
    if (Number.isNaN(hh) || Number.isNaN(mm)) return null
    return hh * 60 + mm
  }
  const srMin = parse(sunrise)
  const ssMin = parse(sunset)
  if (srMin != null && ssMin != null) {
    return nowMin < srMin || nowMin >= ssMin
  }
  // Fallback — <7 or >=19 counts as night.
  return h < 7 || h >= 19
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
  const night    = isNightNow(weather?.sunrise, weather?.sunset)

  return (
    <div className="hero-block hero-block--weather wc-card">
      <div className="wc-grid">
        <div className="wc-illus" aria-hidden={failed || !hasData ? 'true' : undefined}>
          <WeatherScene icon={icon} night={night} />
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
