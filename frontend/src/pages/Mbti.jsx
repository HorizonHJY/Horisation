import React, { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'
import HandLoader from '../components/HandLoader'

/**
 * /mbti — the MBTI test.
 *
 * Three states:
 *   - 'intro'   : pick 20 or 60 questions (or review existing result)
 *   - 'quiz'    : answering, one statement + two choices at a time
 *   - 'result'  : the four letters + nickname + description + match + redo
 *
 * The quiz keeps answers client-side as the user picks; on the last question
 * it POSTs the whole array to /api/mbti/submit and lets the server score.
 */

const VIS_LABEL = { private: '仅自己', friends: '仅好友', public: '公开' }

function dimRow(label, left, leftN, right, rightN) {
  const total = leftN + rightN
  const pct = total ? Math.round(leftN / total * 100) : 50
  return (
    <div className="mbti-dim" key={label}>
      <div className="mbti-dim__row">
        <span className="mbti-dim__letter">{left}</span>
        <div className="mbti-dim__bar">
          <span className="mbti-dim__fill" style={{ width: `${pct}%` }} />
        </div>
        <span className="mbti-dim__letter mbti-dim__letter--right">{right}</span>
      </div>
      <div className="mbti-dim__nums">
        <span>{leftN}</span><span>{rightN}</span>
      </div>
    </div>
  )
}

// ── Intro ─────────────────────────────────────────────────────────
function Intro({ existing, onStart }) {
  return (
    <section className="mbti-intro">
      <h2 className="mbti-h">MBTI 性格测试</h2>
      <p className="text-muted">
        由 20 道或 60 道题组成,结果是四个字母(例:INFP)。玩玩就好,不是心理学诊断。
      </p>
      {existing && (
        <div className="mbti-existing">
          <p className="small text-muted mb-1">你上次测试结果</p>
          <div className="d-flex align-items-center gap-3">
            <div className="mbti-badge mbti-badge--lg">{existing.type}</div>
            <div>
              <strong>{existing.type_card?.nick}</strong>
              <div className="small text-muted">{existing.type_card?.tag}</div>
            </div>
          </div>
        </div>
      )}
      <div className="mbti-mode mt-4">
        <div className="row g-3">
          <div className="col-md-6">
            <button className="mbti-mode__card" onClick={() => onStart('short')}>
              <div className="mbti-mode__num">20</div>
              <div className="mbti-mode__name">快速版</div>
              <div className="mbti-mode__hint">大概 3 分钟 · 结果 70% 相似</div>
            </button>
          </div>
          <div className="col-md-6">
            <button className="mbti-mode__card" onClick={() => onStart('full')}>
              <div className="mbti-mode__num">60</div>
              <div className="mbti-mode__name">标准版</div>
              <div className="mbti-mode__hint">大概 10 分钟 · 接近 16personalities</div>
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}

// ── Quiz ──────────────────────────────────────────────────────────
function Quiz({ questions, mode, onDone, onCancel }) {
  const [i, setI] = useState(0)
  const [answers, setAnswers] = useState([])
  const [submitting, setSubmitting] = useState(false)

  const total = questions.length
  const q = questions[i]
  const pct = Math.round((i / total) * 100)

  const pick = async (choiceIdx) => {
    const next = [...answers, choiceIdx]
    setAnswers(next)
    if (next.length < total) {
      setI(i + 1)
      return
    }
    setSubmitting(true)
    const d = await api.post('/api/mbti/submit', { mode, answers: next })
    setSubmitting(false)
    if (!d.ok) { alert(d.error || '提交失败,请再试一次'); return }
    onDone(d)
  }

  const back = () => { if (i > 0) { setI(i - 1); setAnswers(answers.slice(0, -1)) } }

  if (submitting) return <div className="text-center py-5"><HandLoader /></div>

  return (
    <section className="mbti-quiz">
      <div className="mbti-progress">
        <div className="mbti-progress__bar" style={{ width: `${pct}%` }} />
        <div className="mbti-progress__label small text-muted">
          题目 {i + 1} / {total}
        </div>
      </div>

      <h3 className="mbti-q__statement">{q.statement}</h3>
      <div className="mbti-choices">
        {q.choices.map((c, idx) => (
          <button key={idx} className="mbti-choice" onClick={() => pick(idx)}>
            <span className="mbti-choice__letter">{String.fromCharCode(65 + idx)}</span>
            <span className="mbti-choice__text">{c.label}</span>
          </button>
        ))}
      </div>

      <div className="mbti-quiz__foot d-flex gap-2 mt-3">
        <button className="btn btn-outline-secondary btn-sm" onClick={back} disabled={i === 0}>
          <i className="fas fa-arrow-left me-1" /> 上一题
        </button>
        <button className="btn btn-link btn-sm text-muted" onClick={onCancel}>
          放弃本次测试
        </button>
      </div>
    </section>
  )
}

// ── Result ────────────────────────────────────────────────────────
function Result({ data, onRetake }) {
  const [vis, setVis] = useState(data.result.visibility)
  const r = data.result, card = data.type_card
  const c = r.counts

  const changeVis = async (v) => {
    setVis(v)
    await api.patch('/api/mbti/me/visibility', { visibility: v })
  }

  return (
    <section className="mbti-result">
      <div className="mbti-result__head">
        <div className="mbti-badge mbti-badge--xl">{r.type}</div>
        <div className="mbti-result__name">
          <h2 className="mbti-h mb-0">{card?.nick}</h2>
          <div className="mbti-result__tag">{card?.tag}</div>
        </div>
      </div>

      <p className="mbti-result__desc">{card?.desc}</p>

      <div className="mbti-dims">
        {dimRow('E/I', 'E', c.E, 'I', c.I)}
        {dimRow('S/N', 'S', c.S, 'N', c.N)}
        {dimRow('T/F', 'T', c.T, 'F', c.F)}
        {dimRow('J/P', 'J', c.J, 'P', c.P)}
      </div>

      <div className="mbti-two">
        <div className="mbti-two__col">
          <h4 className="mbti-sub">优势</h4>
          <ul className="mbti-list">{card?.strengths.map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
        <div className="mbti-two__col">
          <h4 className="mbti-sub">盲点</h4>
          <ul className="mbti-list mbti-list--muted">{card?.blind.map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
      </div>

      <div className="mbti-match">
        <h4 className="mbti-sub">常见搭配</h4>
        <div className="d-flex gap-2 flex-wrap">
          {card?.match.map(t => (
            <span key={t} className="mbti-badge mbti-badge--sm">{t}</span>
          ))}
        </div>
      </div>

      <div className="mbti-vis">
        <h4 className="mbti-sub">可见范围</h4>
        <div className="radio-inputs">
          {['private', 'friends', 'public'].map(v => (
            <label className="radio" key={v}>
              <input type="radio" name="mbti-vis" checked={vis === v} onChange={() => changeVis(v)} />
              <span className="name">{VIS_LABEL[v]}</span>
            </label>
          ))}
        </div>
        <p className="small text-muted mt-1">
          {vis === 'public'  && '任何人都能在你 profile 和聊天里看到你的 MBTI。'}
          {vis === 'friends' && '只有好友能在聊天和你的 profile 看到。'}
          {vis === 'private' && '只有你自己能看到。'}
        </p>
      </div>

      <div className="d-flex gap-2 mt-3">
        <button className="btn btn-primary" onClick={onRetake}>重新测试</button>
        <Link to="/profile" className="btn btn-outline-secondary">去 profile 看看</Link>
      </div>

      <p className="small text-muted mt-4">
        第 {new Date(r.taken_at).toLocaleDateString('zh-CN')} 次测试({r.mode === 'short' ? '20 题' : '60 题'})。
        娱乐性质,不构成心理学诊断。
      </p>
    </section>
  )
}

// ── Page ──────────────────────────────────────────────────────────
export default function Mbti() {
  const { user } = useAuth()
  const [stage, setStage]       = useState('intro')
  const [loading, setLoading]   = useState(true)
  const [existing, setExisting] = useState(null)
  const [questions, setQuestions] = useState([])
  const [mode, setMode]         = useState('short')
  const [resultData, setResultData] = useState(null)

  const loadMe = useCallback(async () => {
    if (!user) { setLoading(false); return }
    const d = await api.get('/api/mbti/me')
    if (d.ok && d.result) {
      setExisting({ ...d.result, type_card: d.type_card })
      setResultData({ result: d.result, type_card: d.type_card })
      setStage('result')
    }
    setLoading(false)
  }, [user])

  useEffect(() => { loadMe() }, [loadMe])

  const start = async (m) => {
    setMode(m); setLoading(true)
    const d = await api.get(`/api/mbti/questions?mode=${m}`)
    setLoading(false)
    if (!d.ok) return
    setQuestions(d.questions)
    setStage('quiz')
  }

  const finish = (data) => {
    setResultData(data)
    setExisting({ ...data.result, type_card: data.type_card })
    setStage('result')
  }

  const retake = () => { setStage('intro'); setResultData(null) }

  if (!user) {
    return (
      <div className="mbti-page">
        <div className="mbti-page__inner">
          <h1 className="mbti-h">MBTI 性格测试</h1>
          <p className="text-muted">请先登录,好把结果存到你的 profile。</p>
          <Link to="/login" className="btn btn-primary">去登录</Link>
        </div>
      </div>
    )
  }

  if (loading) return <div className="mbti-page"><div className="mbti-page__inner text-center py-5"><HandLoader /></div></div>

  return (
    <div className="mbti-page">
      <div className="mbti-page__inner">
        <h1 className="mbti-h mbti-h--title">MBTI<span className="mbti-h__sub">16 Personalities</span></h1>
        {stage === 'intro'  && <Intro existing={existing} onStart={start} />}
        {stage === 'quiz'   && <Quiz questions={questions} mode={mode} onDone={finish} onCancel={() => setStage('intro')} />}
        {stage === 'result' && resultData && <Result data={resultData} onRetake={retake} />}
      </div>
    </div>
  )
}
