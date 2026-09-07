import React, { useEffect, useRef, useCallback, useId } from 'react'

/**
 * Shared modal shell.
 *
 * Every modal in the app used to be a bare div: no Escape, no focus trap, no
 * role, and no scroll lock — so on iOS Safari the page scrolled behind the
 * dialog and closing it dumped you somewhere else in the list. Anything that
 * covers the page goes through here instead.
 */

const FOCUSABLE = [
  'a[href]', 'button:not([disabled])', 'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])', 'textarea:not([disabled])', '[tabindex]:not([tabindex="-1"])',
].join(',')

/** Lock the page behind the modal. Uses position:fixed rather than
 *  overflow:hidden because iOS Safari ignores the latter, then restores the
 *  exact scroll offset so closing never moves the reader. */
function useScrollLock() {
  useEffect(() => {
    const y = window.scrollY
    const { body } = document
    const prev = {
      position: body.style.position,
      top: body.style.top,
      width: body.style.width,
      overflowY: body.style.overflowY,
    }
    body.style.position = 'fixed'
    body.style.top = `-${y}px`
    body.style.width = '100%'
    body.style.overflowY = 'scroll'   // keep the scrollbar gutter, no layout jump

    return () => {
      body.style.position = prev.position
      body.style.top = prev.top
      body.style.width = prev.width
      body.style.overflowY = prev.overflowY
      window.scrollTo(0, y)
    }
  }, [])
}

export default function Modal({
  onClose,
  title,
  titleId: titleIdProp,
  size = '',              // '' | 'modal-lg'
  scrollable = true,
  dismissOnBackdrop = true,
  className = '',
  contentStyle,
  children,
}) {
  const dialogRef = useRef(null)
  const restoreRef = useRef(null)
  const generatedId = useId()
  const titleId = titleIdProp || `modal-title-${generatedId}`

  useScrollLock()

  // Remember what had focus, move focus into the dialog, put it back on close.
  useEffect(() => {
    restoreRef.current = document.activeElement
    const node = dialogRef.current
    if (node) {
      const first = node.querySelector(FOCUSABLE)
      ;(first || node).focus({ preventScroll: true })
    }
    return () => {
      const prev = restoreRef.current
      if (prev && typeof prev.focus === 'function') prev.focus({ preventScroll: true })
    }
  }, [])

  const onKeyDown = useCallback((e) => {
    if (e.key === 'Escape') {
      e.stopPropagation()
      onClose?.()
      return
    }
    if (e.key !== 'Tab') return

    const node = dialogRef.current
    if (!node) return
    const items = Array.from(node.querySelectorAll(FOCUSABLE))
      .filter(el => el.offsetParent !== null || el === document.activeElement)
    if (items.length === 0) {
      e.preventDefault()
      return
    }
    const first = items[0]
    const last = items[items.length - 1]
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault()
      last.focus()
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault()
      first.focus()
    }
  }, [onClose])

  return (
    <div
      className="modal show d-block"
      style={{ background: 'rgba(0,0,0,.5)' }}
      onMouseDown={dismissOnBackdrop ? (e) => { if (e.target === e.currentTarget) onClose?.() } : undefined}
      onKeyDown={onKeyDown}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? titleId : undefined}
        aria-label={title ? undefined : 'Dialog'}
        tabIndex={-1}
        className={`modal-dialog modal-dialog-centered ${size} ${scrollable ? 'modal-dialog-scrollable' : ''} ${className}`}
      >
        <div className="modal-content" style={contentStyle}>
          {typeof children === 'function' ? children({ titleId }) : children}
        </div>
      </div>
    </div>
  )
}

/**
 * Confirmation dialog for destructive actions.
 *
 * Replaces window.confirm, which on iOS Safari renders as an unstyleable
 * "horizonyhj.com says:" sheet and — worse — never named the item being
 * destroyed, because the modal that launched it had already closed.
 */
export function ConfirmDialog({
  title = 'Are you sure?',
  message,
  itemName,
  confirmLabel = 'Delete',
  cancelLabel = 'Cancel',
  destructive = true,
  busy = false,
  onConfirm,
  onClose,
}) {
  return (
    <Modal onClose={busy ? () => {} : onClose} title={title} scrollable={false} dismissOnBackdrop={!busy}>
      {({ titleId }) => (
        <>
          <div className="modal-header">
            <h5 className="modal-title fw-semibold" id={titleId} style={{ fontSize: '1rem' }}>{title}</h5>
            <button type="button" className="btn-close" aria-label="Close" onClick={onClose} disabled={busy} />
          </div>
          <div className="modal-body">
            {itemName && (
              <p className="mb-2" style={{ fontFamily: 'var(--font-display)', fontSize: '1.05rem', fontWeight: 600 }}>
                {itemName}
              </p>
            )}
            <p className="mb-0" style={{ color: 'var(--text-secondary)', fontSize: '.9rem' }}>{message}</p>
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose} disabled={busy}>
              {cancelLabel}
            </button>
            <button
              type="button"
              className={`btn ${destructive ? 'btn-danger' : 'btn-primary'}`}
              onClick={onConfirm}
              disabled={busy}
            >
              {busy && <span className="spinner-border spinner-border-sm me-1" />}
              {confirmLabel}
            </button>
          </div>
        </>
      )}
    </Modal>
  )
}
