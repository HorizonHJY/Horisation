/**
 * FlowerCanvas.jsx
 * Watercolour petal-growth animation adapted from react-app.js.
 * Renders into its container's bounds. Children are overlaid on top.
 */
import React, { useEffect, useRef } from 'react'

export default function FlowerCanvas({ children }) {
  const displayCanvasRef = useRef(null)

  useEffect(() => {
    const displayCanvas = displayCanvasRef.current
    if (!displayCanvas) return

    const FILTER_ID        = 'wc-flowers-' + Math.random().toString(36).slice(2)
    const COMPOSITE_FILTER = `url(#${FILTER_ID}) saturate(0.2) contrast(1.15) brightness(1.05)`
    const STAMEN_COLOR     = { h: 30, s: 10, l: 10 }
    const FLOWER_TYPE_COUNT = 6

    const displayCtx = displayCanvas.getContext('2d')
    const paintCanvas = document.createElement('canvas')
    const paintCtx    = paintCanvas.getContext('2d')
    paintCtx.lineCap  = 'round'
    paintCtx.lineJoin = 'round'

    let width, height, animationId, resizeRafId = null
    let stems = [], flowers = []

    const PALETTE = {
      stems:   [{ h:30,s:5,l:15 }, { h:30,s:5,l:10 }, { h:30,s:5,l:20 }],
      flowers: [
        { h:40,s:15,l:90 }, { h:30,s:20,l:85 }, { h:210,s:5,l:88 },
        { h:40,s:10,l:95 }, { h:30,s:15,l:80 }, { h:0,s:0,l:10 },
      ],
    }

    function pickFlowerType() {
      const r = Math.random()
      if (r < 0.22) return 0
      if (r < 0.36) return 1
      if (r < 0.56) return 2
      if (r < 0.74) return 3
      if (r < 0.90) return 4
      return FLOWER_TYPE_COUNT - 1
    }

    function resize() {
      const container = displayCanvas.parentElement
      width  = container ? container.clientWidth  : 400
      height = container ? container.clientHeight : 600
      displayCanvas.width  = width
      displayCanvas.height = height
      paintCanvas.width    = width
      paintCanvas.height   = height
      paintCtx.lineCap  = 'round'
      paintCtx.lineJoin = 'round'
    }

    function renderFrame() {
      displayCtx.clearRect(0, 0, width, height)
      displayCtx.filter = COMPOSITE_FILTER
      displayCtx.drawImage(paintCanvas, 0, 0)
      displayCtx.filter = 'none'
    }

    function shiftScene(dx, dy) {
      for (const stem of stems) {
        stem.x += dx; stem.y += dy
        stem.tip.x += dx; stem.tip.y += dy
        for (const seg of stem.segments) { seg.x += dx; seg.y += dy }
      }
      for (const f of flowers) { f.x += dx; f.y += dy }
    }

    function preserveOnResize() {
      const prevW = width, prevH = height
      const snapshot = document.createElement('canvas')
      snapshot.width = prevW; snapshot.height = prevH
      snapshot.getContext('2d').drawImage(paintCanvas, 0, 0)
      resize()
      const dx = (width - prevW) * 0.5, dy = (height - prevH) * 0.5
      paintCtx.clearRect(0, 0, width, height)
      paintCtx.drawImage(snapshot, dx, dy)
      shiftScene(dx, dy)
      renderFrame()
    }

    // ── Brush ─────────────────────────────────────────────────────────────────
    class Brush {
      constructor(ctx) { this.ctx = ctx }
      stroke(x1, y1, x2, y2, color, w, opacity) {
        this.ctx.beginPath()
        this.ctx.moveTo(x1, y1)
        this.ctx.lineTo(x2, y2)
        this.ctx.strokeStyle = `hsla(${color.h},${color.s}%,${color.l}%,${Math.min(1, opacity * 1.5)})`
        this.ctx.lineWidth = w * (0.8 + Math.random() * 0.4)
        this.ctx.stroke()
      }
      wash(x, y, radius, color, opacity) {
        this.ctx.beginPath()
        this.ctx.arc(x, y, radius, 0, Math.PI * 2)
        this.ctx.fillStyle = `hsla(${color.h},${color.s}%,${color.l}%,${Math.min(1, opacity * 1.5)})`
        this.ctx.fill()
      }
      blob(x, y, radius, color, opacity, angle, stretch) {
        const a  = (angle  ?? Math.random() * Math.PI) + (Math.random() - 0.5) * 0.15
        const sx = (stretch ?? (1.1 + Math.random() * 0.35)) + (Math.random() - 0.5) * 0.1
        const jx = (Math.random() - 0.5) * radius * 0.15
        const jy = (Math.random() - 0.5) * radius * 0.15
        const r  = Math.max(0.8, radius * (0.85 + Math.random() * 0.3))
        const h  = color.h + (Math.random() * 6 - 3)
        const l  = color.l + (Math.random() * 4 - 2)
        this.ctx.beginPath()
        this.ctx.ellipse(x + jx, y + jy, r * sx, r * (0.7 + Math.random() * 0.3), a, 0, Math.PI * 2)
        this.ctx.fillStyle = `hsla(${h},${color.s}%,${l}%,${Math.min(1, opacity * 1.2)})`
        this.ctx.fill()
      }
    }

    // ── Stem ──────────────────────────────────────────────────────────────────
    class Stem {
      constructor(x, y, stemHeight, angle) {
        this.x = x; this.y = y
        this.currentHeight = 0
        this.segments = []
        this.growSpeed = 1.5 + Math.random() * 2
        this.done = false
        this.leavesBySegment = []
        this.leafDensity = 0.18 + Math.random() * 0.1

        let cx = x, cy = y, ca = angle
        const stepSize   = 5
        const totalSteps = Math.floor(stemHeight / stepSize)

        for (let i = 0; i < totalSteps; i++) {
          const targetAngle = -Math.PI / 2 - 0.2
          ca += (targetAngle - ca) * 0.01 + (Math.random() - 0.5) * 0.15
          const nx = cx + Math.cos(ca) * stepSize
          const ny = cy + Math.sin(ca) * stepSize
          this.segments.push({ x: nx, y: ny, angle: ca, cos: Math.cos(ca), sin: Math.sin(ca) })
          if (i > totalSteps * 0.1 && i < totalSteps * 0.95 && Math.random() < this.leafDensity) {
            const side     = Math.random() < 0.5 ? -1 : 1
            const leafAngle = ca + side * (0.5 + Math.random() * 0.6)
            const leaf      = {
              angle: leafAngle, cos: Math.cos(leafAngle), sin: Math.sin(leafAngle),
              length: 20 + Math.random() * 25, radius: 5 + Math.random() * 5, drawn: false,
            }
            if (!this.leavesBySegment[i]) this.leavesBySegment[i] = []
            this.leavesBySegment[i].push(leaf)
          }
          cx = nx; cy = ny
        }
        this.tip = { x: cx, y: cy }
      }

      update() {
        if (this.currentHeight < this.segments.length) {
          this.currentHeight += this.growSpeed
          if (this.currentHeight >= this.segments.length) {
            this.currentHeight = this.segments.length
            this.done = true
          }
        }
        return this.done
      }

      draw(brush) {
        const maxIndex = Math.floor(this.currentHeight)
        if (maxIndex < 1) return
        const drawEnd  = this.hasFlower ? Math.max(1, maxIndex - 1) : maxIndex
        const startIdx = Math.max(1, drawEnd - Math.ceil(this.growSpeed) - 1)
        const segCount = this.segments.length
        for (let i = startIdx; i < drawEnd; i++) {
          const prev = this.segments[i - 1], curr = this.segments[i]
          const w = 3.5 * (1 - i / segCount) + 0.5
          brush.stroke(prev.x, prev.y, curr.x, curr.y, PALETTE.stems[0], w, 0.9)
          const leaves = this.leavesBySegment[i]
          if (leaves) {
            for (const leaf of leaves) {
              if (!leaf.drawn) { this._drawLeaf(brush, curr.x, curr.y, leaf, w); leaf.drawn = true }
            }
          }
        }
      }

      _drawLeaf(brush, x, y, leaf, stemWidth) {
        const tipX = x + leaf.cos * leaf.length, tipY = y + leaf.sin * leaf.length
        const mid  = leaf.length * 0.55
        const midX = x + leaf.cos * mid, midY = y + leaf.sin * mid
        const nx   = -leaf.sin, ny = leaf.cos
        const lo   = leaf.radius * 0.55
        brush.blob(midX, midY,           leaf.radius * 1.2, PALETTE.stems[0], 0.7, leaf.angle, 2.2)
        brush.blob(midX + nx*lo, midY + ny*lo, leaf.radius*0.9,  PALETTE.stems[1], 0.7, leaf.angle+0.15, 2.0)
        brush.blob(midX - nx*lo, midY - ny*lo, leaf.radius*0.85, PALETTE.stems[2], 0.7, leaf.angle-0.15, 1.9)
        brush.stroke(x - leaf.cos*(leaf.radius*0.35), y - leaf.sin*(leaf.radius*0.35),
                     tipX, tipY, PALETTE.stems[2], Math.max(0.4, stemWidth*0.3), 0.8)
      }
    }

    // ── Flower ────────────────────────────────────────────────────────────────
    class Flower {
      constructor(x, y, type, scale) {
        this.x = x; this.y = y; this.scale = scale || 1
        this.type = type ?? pickFlowerType()
        this.age  = 0; this.maxAge = 100 + Math.random() * 50
        this.petals = []; this.centerDots = []
        const pal = PALETTE.flowers
        this.colorBias  = Math.floor(Math.random() * pal.length)
        this.baseColor  = pal[this.colorBias]
        this.altColor   = pal[(this.colorBias + 2) % pal.length]
        this.darkColor  = { h: this.baseColor.h, s: Math.min(100, this.baseColor.s+10), l: Math.max(15,  this.baseColor.l-20) }
        this.lightColor = { h: this.baseColor.h, s: Math.max(20,  this.baseColor.s-10), l: Math.min(95, this.baseColor.l+25) }
        this.petalDrawChance = 0.7
        this.headTilt = (Math.random() - 0.5) * 0.8
        this.headLift = 12 * this.scale
        this._initShape()
      }

      _addPetal(angle, distance, radius, layer, stretch) {
        this.petals.push({
          angle, cos: Math.cos(angle), sin: Math.sin(angle),
          distance: distance * this.scale, radius: radius * this.scale,
          layer: layer || 0,
          stretch: (stretch ?? 1.45) + (Math.random() - 0.5) * 0.24,
        })
      }

      _addDots(count, minD, maxD, minS, maxS, color) {
        for (let i = 0; i < count; i++) {
          const a = Math.random() * Math.PI * 2
          this.centerDots.push({
            cos: Math.cos(a), sin: Math.sin(a),
            distance: (minD + Math.random() * (maxD - minD)) * this.scale,
            size:     (minS + Math.random() * (maxS - minS)) * this.scale,
            color:    color || STAMEN_COLOR,
          })
        }
      }

      _initShape() {
        if (this.type === 0) {
          this.petalDrawChance = 0.65
          for (let layer = 0; layer < 3; layer++) {
            const count = 4 + layer * 2
            for (let i = 0; i < count; i++) {
              const a = (Math.PI*2/count)*i + (Math.random()-0.5)*0.5
              this._addPetal(a, 5+layer*4, 9+layer*3, layer, 1.2)
            }
          }
          this._addDots(15, 1, 3, 1.5, 2.5, { h:0,s:0,l:10 })
        } else if (this.type === 1) {
          const base = -Math.PI / 2
          this._addPetal(base-0.4, 12, 12, 1, 1.4)
          this._addPetal(base+0.4, 12, 12, 1, 1.4)
          this._addPetal(base,     14, 14, 0, 1.2)
          this._addDots(6, 1, 4, 1, 2, STAMEN_COLOR)
        } else if (this.type === 2) {
          for (let i = 0; i < 12; i++) this._addPetal((Math.PI*2/12)*i, 16, 6, 0, 2.5)
          this._addDots(10, 0, 4, 1.5, 2.5, STAMEN_COLOR)
        } else if (this.type === 3) {
          for (let layer = 0; layer < 5; layer++) {
            const count = 5 + layer * 3
            for (let i = 0; i < count; i++) this._addPetal((Math.PI*2/count)*i+layer, 2+layer*3, 5+layer, layer, 1.1)
          }
        } else {
          const count = 7, off = -Math.PI / 2
          for (let i = 0; i < count; i++) {
            const a = off + (Math.PI/count)*(i-count/2)*0.8 + (Math.random()-0.5)*0.3
            this._addPetal(a, 15, 8, 0, 1.8)
          }
          this._addDots(5, 1, 3, 1, 2, STAMEN_COLOR)
        }
      }

      draw(brush) {
        if (this.age > this.maxAge) return
        this.age++
        const growth = this.age < 60 ? this.age / 60 : 1
        const bloom  = 0.2 + growth * 0.8
        const headX  = this.x + this.headTilt * this.headLift * 0.5 * bloom
        const headY  = this.y - this.headLift * bloom
        for (const p of this.petals) {
          if (Math.random() > this.petalDrawChance) continue
          const dist = p.distance * bloom, radius = p.radius * bloom
          let color = this.baseColor
          if (p.layer === 0)  color = this.darkColor
          else if (p.layer > 2) color = this.lightColor
          brush.blob(headX + p.cos*dist, headY + p.sin*dist, radius, color, 0.4, p.angle, p.stretch)
        }
        if (this.age > 15) {
          const dg = Math.min(1, (this.age - 15) / 30)
          for (const d of this.centerDots) {
            if (Math.random() > 0.4)
              brush.wash(headX + d.cos*d.distance*dg, headY + d.sin*d.distance*dg, d.size*dg, d.color, 0.9)
          }
        }
      }
    }

    // ── Main loop ─────────────────────────────────────────────────────────────
    let brush

    function init() {
      cancelAnimationFrame(animationId)
      resize()
      paintCtx.clearRect(0, 0, width, height)
      renderFrame()
      brush   = new Brush(paintCtx)
      stems   = []
      flowers = []

      const startY = height + 40
      const count  = Math.max(8, Math.floor(width / 22))
      for (let i = 0; i < count; i++) {
        const startX = width * 0.05 + Math.random() * width * 0.9
        const angle  = -Math.PI/2 - 0.2 + (Math.random() - 0.5) * 0.6
        const h      = height * 0.35 + Math.random() * (height * 0.55)
        const stem   = new Stem(startX, startY, h, angle)
        stem.flowerTypeHint = pickFlowerType()
        stems.push(stem)
      }
      loop()
    }

    function loop() {
      let allStemsDone = true
      for (const stem of stems) {
        const done = stem.update()
        stem.draw(brush)
        if (!done) {
          allStemsDone = false
        } else if (!stem.hasFlower) {
          const anchorIdx = Math.max(1, stem.segments.length - 2)
          const anchor    = stem.segments[anchorIdx]
          flowers.push(new Flower(anchor.x, anchor.y, stem.flowerTypeHint, 0.5 + Math.random() * 0.4))
          if (stem.segments.length > 20 && Math.random() > 0.5) {
            const budNode = stem.segments[Math.floor(stem.segments.length * 0.7)]
            flowers.push(new Flower(budNode.x + (Math.random()-0.5)*20, budNode.y, pickFlowerType(), 0.3+Math.random()*0.2))
          }
          stem.hasFlower = true
        }
      }
      let allFlowersDone = true
      for (const f of flowers) {
        f.draw(brush)
        if (f.age < f.maxAge) allFlowersDone = false
      }
      renderFrame()
      if (!allStemsDone || !allFlowersDone) {
        animationId = requestAnimationFrame(loop)
      } else {
        animationId = null
      }
    }

    // Inject watercolor SVG filter
    const svgNS = 'http://www.w3.org/2000/svg'
    const svg   = document.createElementNS(svgNS, 'svg')
    svg.setAttribute('style', 'position:absolute;width:0;height:0')
    const defs   = document.createElementNS(svgNS, 'defs')
    const filter = document.createElementNS(svgNS, 'filter')
    filter.setAttribute('id', FILTER_ID)
    filter.innerHTML = `
      <feTurbulence type="fractalNoise" baseFrequency="0.01" numOctaves="1" result="noise"/>
      <feDisplacementMap in="SourceGraphic" in2="noise" scale="8"/>
      <feGaussianBlur stdDeviation="0.4"/>
      <feComponentTransfer><feFuncA type="linear" slope="1.2"/></feComponentTransfer>`
    defs.appendChild(filter)
    svg.appendChild(defs)
    document.body.appendChild(svg)

    const ro = new ResizeObserver(() => {
      if (resizeRafId !== null) cancelAnimationFrame(resizeRafId)
      resizeRafId = requestAnimationFrame(() => { resizeRafId = null; preserveOnResize() })
    })
    const container = displayCanvas.parentElement
    if (container) ro.observe(container)

    const timer = setTimeout(init, 300)

    return () => {
      cancelAnimationFrame(animationId)
      if (resizeRafId !== null) cancelAnimationFrame(resizeRafId)
      ro.disconnect()
      clearTimeout(timer)
      svg.remove()
    }
  }, [])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#efebe6', overflow: 'hidden' }}>
      <canvas
        ref={displayCanvasRef}
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
      />
      <div style={{
        position: 'absolute', inset: 0, zIndex: 10,
        display: 'flex', flexDirection: 'column',
        justifyContent: 'center', alignItems: 'center',
        padding: '2.5rem',
      }}>
        {children}
      </div>
    </div>
  )
}
