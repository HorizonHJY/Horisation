const BASE = ''  // same origin; Vite dev proxy handles /api → Flask

async function request(path, options = {}) {
  const res = await fetch(BASE + path, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  return res.json()
}

export const api = {
  get:    (path)         => request(path),
  post:   (path, body)   => request(path, { method: 'POST',   body: JSON.stringify(body) }),
  put:    (path, body)   => request(path, { method: 'PUT',    body: JSON.stringify(body) }),
  delete: (path)         => request(path, { method: 'DELETE' }),

  // Multipart (file upload) — no JSON header
  upload: (path, formData) =>
    fetch(BASE + path, { method: 'POST', credentials: 'include', body: formData }).then(r => r.json()),
}
