import React, { useState, useEffect, useCallback } from 'react'

const ICON_MAP = {
  'fa-sun':              '\u2600\ufe0f',
  'fa-cloud-sun':        '\u26c5',
  'fa-cloud':            '\u2601\ufe0f',
  'fa-cloud-rain':       '\ud83c\udf27\ufe0f',
  'fa-cloud-showers-heavy': '\ud83c\udf27\ufe0f',
  'fa-snowflake':        '\u2744\ufe0f',
  'fa-smog':             '\ud83c\udf2b\ufe0f',
  'fa-bolt':             '\u26a1',
}

function wmoEmoji(icon) {
  return ICON_MAP[icon] || '\u2600\ufe0f'
}

export default function WeatherGreeting({ name }) {
  const [weather, setWeather] = useState(null)

  const fetchWeather = useCallback(async () => {
    try {
      const resp = await fetch('/api/weather', { credentials: 'include' })
      const data = await resp.json()
      if (data.ok) {
        setWeather(data)
      }
    } catch {
      // Silently fail — weather is non-critical
    }
  }, [])

  useEffect(() => {
    fetchWeather()
    const interval = setInterval(fetchWeather, 10 * 60 * 1000)
    return () => clearInterval(interval)
  }, [fetchWeather])

  function todayLabel() {
    // Mon \u00b7 Oct 18 \u2014 short enough to live on the same line as the weather.
    return new Date().toLocaleDateString('en-US',
      { weekday: 'short', month: 'short', day: 'numeric' })
  }

  function greetingLine() {
    if (!weather || weather.temp == null) {
      const h = new Date().getHours()
      const greet = h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening'
      return { text: `${greet}, ${name}`, sub: todayLabel() }
    }

    const greet = weather.greeting.replace('{name}', name)
    const emoji = wmoEmoji(weather.icon)
    // Both units: the readers are Chinese friends living in the US, and
    // neither scale is "the" one at that table.
    const t = `${weather.temp_f}\u00b0F / ${weather.temp_c}\u00b0C`
    const f = `${weather.feels_like_f}\u00b0F / ${weather.feels_like_c}\u00b0C`
    const parts = [
      todayLabel(),
      `${weather.city} \u00b7 ${emoji} ${t} ${weather.label}`,
      `Feels like ${f}`,
    ]
    const tmr = weather.tomorrow
    if (tmr) {
      const tEmoji = wmoEmoji(tmr.icon)
      const hi = `${tmr.high_f}\u00b0F / ${tmr.high_c}\u00b0C`
      const lo = (tmr.low_f != null) ? ` \u2193 ${tmr.low_f}\u00b0F / ${tmr.low_c}\u00b0C` : ''
      parts.push(`Tomorrow ${tEmoji} ${hi}${lo} ${tmr.label}`)
    }
    return { text: greet, sub: parts.join(' \u00b7 ') }
  }

  const line = greetingLine()

  return (
    <div className="hero-block" style={{ paddingBottom: 10 }}>
      <h2 className="hero-title" style={{ marginBottom: 2 }}>
        {line.text}
      </h2>
      {line.sub && (
        <p className="hero-sub" style={{ marginBottom: 0, fontSize: '.95rem' }}>
          {line.sub}
        </p>
      )}
    </div>
  )
}
