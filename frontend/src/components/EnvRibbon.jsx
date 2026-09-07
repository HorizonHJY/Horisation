import React, { useEffect } from 'react'

/**
 * Marks a non-production instance.
 *
 * Local and production both answer on a localhost URL during development, so
 * nothing on screen distinguished them — it was possible to sign in to one and
 * expect the other. This makes the difference impossible to miss, on every
 * page including login, and leaves production completely untouched.
 *
 * Two independent signals, either one is enough:
 *   - import.meta.env.DEV — the Vite dev server is serving this bundle
 *   - local_dev from /api/auth/check-session — Flask started with LOCAL_DEV=1,
 *     which also covers scripts/build-run.bat, where Vite is not involved
 */
export const IS_VITE_DEV = Boolean(import.meta.env?.DEV)

export default function EnvRibbon({ backendLocal }) {
  const isLocal = IS_VITE_DEV || Boolean(backendLocal)

  // The browser tab is often the only thing visible when several windows are
  // open — that is exactly when the two instances get confused.
  useEffect(() => {
    const base = 'Horisation'
    document.title = isLocal ? `[LOCAL] ${base}` : base
  }, [isLocal])

  if (!isLocal) return null

  const source = IS_VITE_DEV ? 'vite dev server' : 'Flask LOCAL_DEV=1'

  return (
    <div className="env-ribbon" aria-hidden="true" data-testid="env-ribbon">
      <div className="env-ribbon__bar" />
      <div className="env-ribbon__tag" title={`Local instance — ${source}`}>
        <i className="fas fa-flask" />
        <span>LOCAL</span>
      </div>
    </div>
  )
}
