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
    // Mon · Oct 18 — short enough to live on the same line as the weather.
    return new Date().toLocaleDateString('en-US',
      { weekday: 'short', month: 'short', day: 'numeric' })
  }

  function build() {
    // Fallback: no weather yet — at least show the date so the subtitle isn't blank.
    if (!weather || weather.temp == null) {
      const h = new Date().getHours()
      const greet = h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening'
      return { text: `${greet}, ${name}`, today: todayLabel(), weather: null, tomorrow: null }
    }
    const greet = weather.greeting.replace('{name}', name)
    const t = `${weather.temp_f}°F / ${weather.temp_c}°C`
    const f = `${weather.feels_like_f}°F / ${weather.feels_like_c}°C`
    return {
      text: greet,
      today: todayLabel(),
      weather: {
        emoji: wmoEmoji(weather.icon),
        city: weather.city, temp: t, label: weather.label, feels: f,
      },
      tomorrow: weather.tomorrow && {
        emoji: wmoEmoji(weather.tomorrow.icon),
        hi: `${weather.tomorrow.high_f}°F / ${weather.tomorrow.high_c}°C`,
        lo: weather.tomorrow.low_f != null
          ? `${weather.tomorrow.low_f}°F / ${weather.tomorrow.low_c}°C`
          : null,
        label: weather.tomorrow.label,
      },
    }
  }

  const g = build()

  return (
    <div className="hero-block hero-block--weather" style={{ paddingBottom: 10 }}>
      <h2 className="hero-title" style={{ marginBottom: 2 }}>
        {g.text}
      </h2>
      {/* Today — split in two rows: the when / where / what, then the numbers.
          Horizon 2026-10-19: single-line felt too packed on phones. */}
      <p className="hero-sub hero-sub--today">
        <span className="wg-date">{g.today}</span>
        {g.weather && (
          <>
            {' · '}
            <span className="wg-city">{g.weather.city}</span>
            {' '}<span className="wg-emoji" aria-hidden="true">{g.weather.emoji}</span>
            {' '}<span className="wg-label">{g.weather.label}</span>
          </>
        )}
      </p>
      {g.weather && (
        <p className="hero-sub hero-sub--today-temp">
          <span className="wg-temp">{g.weather.temp}</span>
          {' · '}
          <span className="wg-feels">Feels like {g.weather.feels}</span>
        </p>
      )}
      {/* Tomorrow — split in two rows, same shape as today. Row 1 is label +
          emoji + condition; row 2 is the HIGH / LOW numbers (Horizon 2026-10-19). */}
      {g.tomorrow && (
        <>
          <p className="hero-sub hero-sub--tomorrow">
            <span className="wg-tomorrow-label">Tomorrow</span>
            {' '}<span aria-hidden="true">{g.tomorrow.emoji}</span>{' '}
            <span className="wg-label">{g.tomorrow.label}</span>
          </p>
          <p className="hero-sub hero-sub--tomorrow-temp">
            <span className="wg-hi">High</span>{' '}
            <span className="wg-temp">{g.tomorrow.hi}</span>
            {g.tomorrow.lo && (
              <>
                {' · '}
                <span className="wg-lo">Low</span>{' '}
                <span className="wg-temp">{g.tomorrow.lo}</span>
              </>
            )}
          </p>
        </>
      )}
    </div>
  )
}
