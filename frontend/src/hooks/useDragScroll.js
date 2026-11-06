import { useEffect } from 'react'

const NODRAG_SELECTOR =
  'input, textarea, select, [contenteditable], [draggable="true"], [data-nodrag]'

const THRESHOLD_PX = 5
const FRICTION = 0.94
const MIN_V = 0.05

function isInNoDragElement(el) {
  return !!(el && el.closest && el.closest(NODRAG_SELECTOR))
}

function hasBlockingScrollableAncestor(el) {
  let cur = el
  while (cur && cur.nodeType === 1 && cur !== document.body && cur !== document.documentElement) {
    const cs = window.getComputedStyle(cur)
    const oy = cs.overflowY
    if ((oy === 'auto' || oy === 'scroll') && cur.scrollHeight > cur.clientHeight) return true
    const ox = cs.overflowX
    if ((ox === 'auto' || ox === 'scroll') && cur.scrollWidth > cur.clientWidth) return true
    cur = cur.parentNode
  }
  return false
}

export default function useDragScroll() {
  useEffect(() => {
    if (typeof window === 'undefined') return

    let state = 'idle'
    let activePointerId = null
    let startX = 0
    let startY = 0
    let lastY = 0
    let lastT = 0
    let pendingDy = 0
    let rafId = 0
    let velocity = 0
    let inertiaRafId = 0
    let inertiaLastT = 0
    let blockClick = false

    const flushScroll = () => {
      rafId = 0
      if (pendingDy !== 0) {
        window.scrollBy(0, pendingDy)
        pendingDy = 0
      }
    }
    const requestFlush = () => {
      if (!rafId) rafId = requestAnimationFrame(flushScroll)
    }

    const stopInertia = () => {
      if (inertiaRafId) {
        cancelAnimationFrame(inertiaRafId)
        inertiaRafId = 0
      }
      velocity = 0
      inertiaLastT = 0
    }

    const inertiaStep = (t) => {
      if (!inertiaLastT) inertiaLastT = t
      const dt = Math.min(t - inertiaLastT, 48)
      inertiaLastT = t
      if (dt > 0) window.scrollBy(0, velocity * dt)
      velocity *= Math.pow(FRICTION, dt / 16)
      if (Math.abs(velocity) < MIN_V) {
        inertiaRafId = 0
        velocity = 0
        return
      }
      inertiaRafId = requestAnimationFrame(inertiaStep)
    }

    const cleanupDragStyles = () => {
      document.body.style.userSelect = ''
      document.body.style.cursor = ''
    }

    const onPointerDown = (e) => {
      blockClick = false
      if (e.button !== 0) return
      if (e.pointerType !== 'mouse' && e.pointerType !== 'pen') return
      const target = e.target
      if (isInNoDragElement(target)) return
      if (hasBlockingScrollableAncestor(target)) return
      stopInertia()
      state = 'pending'
      activePointerId = e.pointerId
      startX = e.clientX
      startY = e.clientY
      lastY = e.clientY
      lastT = e.timeStamp || performance.now()
    }

    const onPointerMove = (e) => {
      if (state === 'idle') return
      if (e.pointerId !== activePointerId) return
      const dx = e.clientX - startX
      const dy = e.clientY - startY
      if (state === 'pending') {
        if (Math.hypot(dx, dy) < THRESHOLD_PX) return
        state = 'dragging'
        document.body.style.userSelect = 'none'
        document.body.style.cursor = 'grabbing'
        const sel = window.getSelection && window.getSelection()
        if (sel && sel.removeAllRanges) sel.removeAllRanges()
        lastY = e.clientY
        lastT = e.timeStamp || performance.now()
        return
      }
      const nowY = e.clientY
      const nowT = e.timeStamp || performance.now()
      const stepDy = lastY - nowY
      pendingDy += stepDy
      const dt = nowT - lastT
      if (dt > 0) {
        const inst = stepDy / dt
        velocity = velocity * 0.6 + inst * 0.4
      }
      lastY = nowY
      lastT = nowT
      requestFlush()
      if (e.cancelable) e.preventDefault()
    }

    const endDrag = (e) => {
      if (state === 'idle') return
      if (e.pointerId !== activePointerId) return
      const wasDragging = state === 'dragging'
      state = 'idle'
      activePointerId = null
      if (wasDragging) {
        cleanupDragStyles()
        blockClick = true
        if (Math.abs(velocity) >= MIN_V) {
          inertiaLastT = 0
          inertiaRafId = requestAnimationFrame(inertiaStep)
        } else {
          velocity = 0
        }
      }
    }

    const onClickCapture = (e) => {
      if (blockClick) {
        blockClick = false
        e.stopPropagation()
        e.preventDefault()
      }
    }

    const onWheel = () => stopInertia()
    const onTouchStart = () => stopInertia()
    const onKeyDown = () => stopInertia()

    window.addEventListener('pointerdown', onPointerDown, true)
    window.addEventListener('pointermove', onPointerMove, { passive: false })
    window.addEventListener('pointerup', endDrag, true)
    window.addEventListener('pointercancel', endDrag, true)
    window.addEventListener('click', onClickCapture, true)
    window.addEventListener('wheel', onWheel, { passive: true })
    window.addEventListener('touchstart', onTouchStart, { passive: true })
    window.addEventListener('keydown', onKeyDown, true)

    return () => {
      window.removeEventListener('pointerdown', onPointerDown, true)
      window.removeEventListener('pointermove', onPointerMove, { passive: false })
      window.removeEventListener('pointerup', endDrag, true)
      window.removeEventListener('pointercancel', endDrag, true)
      window.removeEventListener('click', onClickCapture, true)
      window.removeEventListener('wheel', onWheel)
      window.removeEventListener('touchstart', onTouchStart)
      window.removeEventListener('keydown', onKeyDown, true)
      if (rafId) cancelAnimationFrame(rafId)
      stopInertia()
      cleanupDragStyles()
    }
  }, [])
}
