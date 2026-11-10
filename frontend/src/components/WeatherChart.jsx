import React, { useEffect, useRef, useState } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-basic-dist'

/**
 * 30-day rolling temperature chart for the home page.
 * One API call → double line (hi red, lo blue), dual-unit tooltip, a dotted
 * "today" marker so forecast reads differently from history.
 *
 * basic-dist cuts bundle ~400kB → ~130kB; we don't need maps or heatmaps.
 */

const Plot = createPlotlyComponent(Plotly)

const RED  = '#dc3545'
const BLUE = '#0d6efd'
const MUTED = '#94a3b8'

const UNIT_KEY = 'archbay.weather.unit'

// WMO weather code → symbol + short Chinese label.
function wmoInfo(code) {
  const c = Number(code)
  if (c === 0) return { sym: '☀️', label: '晴' }
  if (c === 1) return { sym: '⛅', label: '多云' }
  if (c === 2 || c === 3) return { sym: '☁️', label: '阴' }
  if (c === 45 || c === 48) return { sym: '🌫️', label: '雾' }
  if (c >= 51 && c <= 57) return { sym: '🌦️', label: '小雨' }
  if (c >= 61 && c <= 67) return { sym: '🌧️', label: '雨' }
  if (c >= 71 && c <= 77) return { sym: '❄️', label: '雪' }
  if (c >= 80 && c <= 82) return { sym: '🌧️', label: '阵雨' }
  if (c === 85 || c === 86) return { sym: '🌨️', label: '雪' }
  if (c >= 95 && c <= 99) return { sym: '⛈️', label: '雷阵雨' }
  return { sym: '·', label: '—' }
}

function readUnit() {
  try {
    const v = localStorage.getItem(UNIT_KEY)
    return v === 'f' ? 'f' : 'c'
  } catch { return 'c' }
}

export default function WeatherChart() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')
  const [unit, setUnit] = useState(readUnit)
  const wrapRef = useRef(null)
  const [width, setWidth] = useState(0)

  useEffect(() => {
    fetch('/api/weather/history', { credentials: 'include' })
      .then(r => r.json())
      .then(d => d.ok ? setData(d) : setErr(d.error || 'Failed to load'))
      .catch(e => setErr(String(e)))
  }, [])

  useEffect(() => {
    try { localStorage.setItem(UNIT_KEY, unit) } catch {}
  }, [unit])

  // Keep the chart responsive to its wrapper width — Plotly's `autosize` works,
  // but only re-measures on window.resize, not on sidebar toggles.
  useEffect(() => {
    if (!wrapRef.current) return
    const ro = new ResizeObserver(entries => {
      for (const e of entries) setWidth(Math.round(e.contentRect.width))
    })
    ro.observe(wrapRef.current)
    return () => ro.disconnect()
  }, [])

  if (err) return <div className="small text-muted py-3">天气图加载失败。</div>
  if (!data) return <div className="small text-muted py-3">Loading chart…</div>

  const pts = data.points || []
  const dates = pts.map(p => p.date)
  const useC = unit === 'c'
  const hi = pts.map(p => useC ? p.hi_c : p.hi_f)
  const lo = pts.map(p => useC ? p.lo_c : p.lo_f)
  const suffix = useC ? '°C' : '°F'

  // Tooltip text with both units.
  const hiText = pts.map(p => `${p.hi_c}°C / ${p.hi_f}°F${p.is_forecast ? ' (forecast)' : ''}`)
  const loText = pts.map(p => `${p.lo_c}°C / ${p.lo_f}°F${p.is_forecast ? ' (forecast)' : ''}`)

  // Forecast portion gets dashed line; history is solid. Plotly can't dash
  // part of a single trace, so split: solid history + dashed forecast, with
  // today appearing in both so the line joins visually.
  const todayIdx = pts.findIndex(p => p.is_today)
  const splitAt  = todayIdx >= 0 ? todayIdx + 1 : pts.length  // include today in history

  const traces = [
    { // History HIGH
      x: dates.slice(0, splitAt), y: hi.slice(0, splitAt), text: hiText.slice(0, splitAt),
      name: 'High', type: 'scatter', mode: 'lines+markers',
      line: { color: RED, width: 2 }, marker: { size: 5, color: RED },
      hovertemplate: '%{x}<br>High: %{text}<extra></extra>',
    },
    { // History LOW
      x: dates.slice(0, splitAt), y: lo.slice(0, splitAt), text: loText.slice(0, splitAt),
      name: 'Low', type: 'scatter', mode: 'lines+markers',
      line: { color: BLUE, width: 2 }, marker: { size: 5, color: BLUE },
      hovertemplate: '%{x}<br>Low: %{text}<extra></extra>',
    },
  ]

  if (splitAt < pts.length) {
    // Overlap the join point so the forecast line connects to today.
    const fStart = Math.max(0, splitAt - 1)
    traces.push(
      { // Forecast HIGH
        x: dates.slice(fStart), y: hi.slice(fStart), text: hiText.slice(fStart),
        name: 'High (fcst)', type: 'scatter', mode: 'lines+markers',
        line: { color: RED, width: 2, dash: 'dot' }, marker: { size: 5, color: RED, symbol: 'diamond' },
        showlegend: false,
        hovertemplate: '%{x}<br>High: %{text}<extra></extra>',
      },
      { // Forecast LOW
        x: dates.slice(fStart), y: lo.slice(fStart), text: loText.slice(fStart),
        name: 'Low (fcst)', type: 'scatter', mode: 'lines+markers',
        line: { color: BLUE, width: 2, dash: 'dot' }, marker: { size: 5, color: BLUE, symbol: 'diamond' },
        showlegend: false,
        hovertemplate: '%{x}<br>Low: %{text}<extra></extra>',
      },
    )
  }

  const layout = {
    autosize: true,
    height: 260,
    margin: { l: 36, r: 12, t: 8, b: 32 },
    showlegend: true,
    legend: { orientation: 'h', y: 1.12, x: 0, bgcolor: 'rgba(0,0,0,0)' },
    hovermode: 'x unified',
    plot_bgcolor:  'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    xaxis: {
      type: 'date', tickformat: '%b %-d',
      gridcolor: 'rgba(0,0,0,.05)', zeroline: false,
      tickfont: { size: 11, color: MUTED },
    },
    yaxis: {
      ticksuffix: suffix,
      title: { text: suffix, font: { size: 11, color: MUTED } },
      gridcolor: 'rgba(0,0,0,.08)', zeroline: false,
      tickfont: { size: 11, color: MUTED },
    },
    shapes: todayIdx >= 0 ? [{
      type: 'line', xref: 'x', yref: 'paper',
      x0: dates[todayIdx], x1: dates[todayIdx], y0: 0, y1: 1,
      line: { color: MUTED, width: 1, dash: 'dot' },
    }] : [],
    annotations: todayIdx >= 0 ? [{
      x: dates[todayIdx], y: 1, xref: 'x', yref: 'paper',
      text: 'today', showarrow: false, font: { size: 10, color: MUTED },
      yshift: 10,
    }] : [],
  }

  const config = {
    displayModeBar: false,
    responsive: true,
    staticPlot: false,
  }

  return (
    <div ref={wrapRef} className="weather-chart">
      <div className="weather-chart-head">
        <div className="weather-unit-toggle" role="group" aria-label="Temperature unit">
          <button type="button"
                  className={`weather-unit-btn${useC ? ' is-active' : ''}`}
                  aria-pressed={useC}
                  onClick={() => setUnit('c')}>°C</button>
          <button type="button"
                  className={`weather-unit-btn${!useC ? ' is-active' : ''}`}
                  aria-pressed={!useC}
                  onClick={() => setUnit('f')}>°F</button>
        </div>
      </div>
      <Plot data={traces} layout={layout} config={config} useResizeHandler
            style={{ width: '100%', height: 260 }} />
      <div className="weather-strip" aria-label="每日天气" style={{ marginLeft: 36, marginRight: 12 }}>
        {pts.map((p, i) => {
          const info = wmoInfo(p.wmo)
          const tip = `${p.date} ${info.label}${p.is_forecast ? '（预报）' : ''}`
          const cls = [
            'weather-strip-cell',
            p.is_today ? 'is-today' : '',
            p.is_forecast ? 'is-forecast' : '',
            i % 2 === 1 ? 'is-alt' : '',
          ].filter(Boolean).join(' ')
          return (
            <div key={p.date} className={cls} title={tip} aria-label={tip}>
              <span className="weather-strip-sym">{info.sym}</span>
              {p.is_forecast && <span className="weather-strip-dot" aria-hidden="true" />}
            </div>
          )
        })}
      </div>
      {width > 0 && <span className="visually-hidden">{width}px</span>}
    </div>
  )
}
