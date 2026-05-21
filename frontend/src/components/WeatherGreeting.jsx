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

  function greetingLine() {
    if (!weather || weather.temp == null) {
      const h = new Date().getHours()
      const greet = h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening'
      return { text: `${greet}, ${name}`, sub: '' }
    }

    const greet = weather.greeting.replace('{name}', name)
    const emoji = wmoEmoji(weather.icon)
    const sub = `${weather.city} \u00b7 ${emoji} ${weather.temp}\u00b0F ${weather.label} \u00b7 Feels like ${weather.feels_like}\u00b0F`
    return { text: greet, sub }
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
