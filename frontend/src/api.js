const BASE = ''  // same origin; Vite dev proxy handles /api → Flask

// Normalize the response body. Never throws on HTTP/network problems so every
// caller can rely on the `data.ok` contract. Only malformed-catastrophe is kept.
async function parseBody(res) {
  const text = await res.text()
  if (!text) return {}                                   // 204 / empty
  try { return JSON.parse(text) } catch { return {} }    // HTML error page etc.
}

// Build a conventional result: on failure always carry .ok=false + readable .error
function failure(message, status, extra = {}) {
  const out = { ok: false, error: message || 'Request failed', ...extra }
  if (status) out.status = status
  return out
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(BASE + path, {
      credentials: 'include',
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    })
  } catch {
    // Network down / connection refused — resolve so callers can show UI error.
    return failure('Network error — check your connection')
  }

  try {
    const body = await parseBody(res)
    if (!res.ok) {
      // Backend error: prefer its message, fall back to HTTP status.
      return failure(body.error || `HTTP ${res.status}`, res.status)
    }
    // Some endpoints only set .ok in the body; ensure it's present & serializable.
    return body && typeof body === 'object' ? { ok: body.ok !== false, ...body } : { ok: true, data: body }
  } catch {
    return failure(`HTTP ${res.status}`)
  }
}

export const api = {
  get:    (path)         => request(path),
  post:   (path, body)   => request(path, { method: 'POST',   body: JSON.stringify(body) }),
  put:    (path, body)   => request(path, { method: 'PUT',    body: JSON.stringify(body) }),
  delete: (path)         => request(path, { method: 'DELETE' }),

  // Multipart (file upload) — no JSON header
  upload: async (path, formData) => {
    let res
    try {
      res = await fetch(BASE + path, { method: 'POST', credentials: 'include', body: formData })
    } catch {
      return failure('Network error — check your connection')
    }
    try {
      const body = await parseBody(res)
      if (!res.ok) return failure(body.error || `HTTP ${res.status}`, res.status)
      return body && typeof body === 'object' ? { ok: body.ok !== false, ...body } : { ok: true, data: body }
    } catch {
      return failure(`HTTP ${res.status}`)
    }
  },
}
