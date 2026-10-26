import React, { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'
import HandLoader from '../components/HandLoader'

/**
 * /mbti — the MBTI test. Bilingual: 中文 / English, toggled on the page and
 * defaulted from the browser language, so an English-speaking friend lands
 * on English without touching anything.
 *
 * Three states:
 *   - 'intro'   : pick 20 or 60 questions (or review an existing result)
 *   - 'quiz'    : one statement + two choices at a time
 *   - 'result'  : four letters + nickname + description + match + redo
 */

const t = (lang, zh, en) => (lang === 'en' ? en : zh)

function dimRow(label, left, leftN, right, rightN) {
  const total = leftN + rightN
  const pct = total ? Math.round(leftN / total * 100) : 50
  return (
    <div className='mbti-dim' key={label}>
      <div className='mbti-dim__row'>
        <span className='mbti-dim__letter'>{left}</span>
        <div className='mbti-dim__bar'>
          <span className='mbti-dim__fill' style={{ width: pct + '%' }} />
        </div>
        <span className='mbti-dim__letter mbti-dim__letter--right'>{right}</span>
      </div>
      <div className='mbti-dim__nums'>
        <span>{leftN}</span><span>{rightN}</span>
      </div>
    </div>
  )
}

// ── Intro ─────────────────────────────────────────────────────────
function Intro({ existing, onStart, lang }) {
  return (
    <section className='mbti-intro'>
      <h2 className='mbti-h'>{t(lang, 'MBTI 性格测试', 'MBTI Personality Test')}</h2>
      <p className='text-muted'>
        {t(lang,
          '由 20 道或 60 道题组成,结果是四个字母(例:INFP)。玩玩就好,不是心理学诊断。',
          'Twenty or sixty questions, scored to four letters (e.g. INFP). Just for fun — not a diagnosis.')}
      </p>
      {existing && (
        <div className='mbti-existing'>
          <p className='small text-muted mb-1'>{t(lang, '你上次测试结果', 'Your last result')}</p>
          <div className='d-flex align-items-center gap-3'>
            <div className='mbti-badge mbti-badge--lg'>{existing.type}</div>
            <div>
              <strong>{lang === 'en' ? existing.type_card?.nick_en : existing.type_card?.nick}</strong>
              <div className='small text-muted'>{lang === 'en' ? existing.type_card?.tag_en : existing.type_card?.tag}</div>
            </div>
          </div>
        </div>
      )}
      <div className='mbti-mode mt-4'>
        <div className='row g-3'>
          <div className='col-md-6'>
            <button className='mbti-mode__card' onClick={() => onStart('short')}>
              <div className='mbti-mode__num'>20</div>
              <div className='mbti-mode__name'>{t(lang, '快速版', 'Quick')}</div>
              <div className='mbti-mode__hint'>{t(lang, '大概 3 分钟 结果 70% 相似', 'About 3 min ~70% match')}</div>
            </button>
          </div>
          <div className='col-md-6'>
            <button className='mbti-mode__card' onClick={() => onStart('full')}>
              <div className='mbti-mode__num'>60</div>
              <div className='mbti-mode__name'>{t(lang, '标准版', 'Standard')}</div>
              <div className='mbti-mode__hint'>{t(lang, '大概 10 分钟 接近 16personalities', 'About 10 min close to 16personalities')}</div>
            </button>
          </div>
        </div>
      </div>

      <div className='mbti-chart mt-4'>
        <h3 className='mbti-sub'>{t(lang, '16 种类型一览', 'The 16 types at a glance')}</h3>
        <img
          src='/mbti-types.jpg'
          alt={t(lang, '16 种 MBTI 类型总览', 'Overview of the 16 MBTI types')}
          className='mbti-chart__img'
          loading='lazy'
        />
      </div>
    </section>
  )
}

// ── Quiz ──────────────────────────────────────────────────────────
function Quiz({ questions, mode, onDone, onCancel, lang }) {
  const [i, setI] = useState(0)
  const [answers, setAnswers] = useState([])
  const [submitting, setSubmitting] = useState(false)

  const total = questions.length
  const q = questions[i]
  const pct = Math.round((i / total) * 100)

  const pickChoice = async (choiceIdx) => {
    const next = [...answers, choiceIdx]
    setAnswers(next)
    if (next.length < total) {
      setI(i + 1)
      return
    }
    setSubmitting(true)
    const d = await api.post('/api/mbti/submit', { mode, answers: next })
    setSubmitting(false)
    if (!d.ok) { alert(d.error || t(lang, '提交失败,请再试一次', 'Could not submit — try again')); return }
    onDone(d)
  }

  const back = () => { if (i > 0) { setI(i - 1); setAnswers(answers.slice(0, -1)) } }

  if (submitting) return <div className='text-center py-5'><HandLoader /></div>

  return (
    <section className='mbti-quiz'>
      <div className='mbti-progress'>
        <div className='mbti-progress__bar' style={{ width: pct + '%' }} />
        <div className='mbti-progress__label small text-muted'>
          {t(lang, '题目', 'Question')} {i + 1} / {total}
        </div>
      </div>

      <h3 className='mbti-q__statement'>{lang === 'en' ? q.statement_en : q.statement}</h3>
      <div className='mbti-choices'>
        {q.choices.map((c, idx) => (
          <button key={idx} className='mbti-choice' onClick={() => pickChoice(idx)}>
            <span className='mbti-choice__letter'>{String.fromCharCode(65 + idx)}</span>
            <span className='mbti-choice__text'>{lang === 'en' ? c.label_en : c.label}</span>
          </button>
        ))}
      </div>

      <div className='mbti-quiz__foot d-flex gap-2 mt-3'>
        <button className='btn btn-outline-secondary btn-sm' onClick={back} disabled={i === 0}>
          <i className='fas fa-arrow-left me-1' /> {t(lang, '上一题', 'Back')}
        </button>
        <button className='btn btn-link btn-sm text-muted' onClick={onCancel}>
          {t(lang, '放弃本次测试', 'Quit test')}
        </button>
      </div>
    </section>
  )
}

// ── Result ────────────────────────────────────────────────────────
function Result({ data, onRetake, lang }) {
  const r = data.result, card = data.type_card
  const c = r.counts

  const en = lang === 'en'

  return (
    <section className='mbti-result'>
      <div className='mbti-result__head'>
        <img src={'/mbti/' + r.type.toLowerCase() + '.png'} alt={r.type} className='mbti-char' />
        <div className='mbti-badge mbti-badge--xl'>{r.type}</div>
        <div className='mbti-result__name'>
          <h2 className='mbti-h mb-0'>{en ? card?.nick_en : card?.nick}</h2>
          <div className='mbti-result__tag'>{en ? card?.tag_en : card?.tag}</div>
        </div>
      </div>

      <p className='mbti-result__desc'>{en ? card?.desc_en : card?.desc}</p>

      <div className='mbti-dims'>
        {dimRow('E/I', 'E', c.E, 'I', c.I)}
        {dimRow('S/N', 'S', c.S, 'N', c.N)}
        {dimRow('T/F', 'T', c.T, 'F', c.F)}
        {dimRow('J/P', 'J', c.J, 'P', c.P)}
      </div>

      <div className='mbti-two'>
        <div className='mbti-two__col'>
          <h4 className='mbti-sub'>{t(lang, '优势', 'Strengths')}</h4>
          <ul className='mbti-list'>{(en ? card?.strengths_en : card?.strengths || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
        <div className='mbti-two__col'>
          <h4 className='mbti-sub'>{t(lang, '盲点', 'Blind spots')}</h4>
          <ul className='mbti-list mbti-list--muted'>{(en ? card?.blind_en : card?.blind || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
      </div>

      <div className='mbti-match'>
        <h4 className='mbti-sub'>{t(lang, '常见搭配', 'Common matches')}</h4>
        <div className='d-flex gap-2 flex-wrap'>
          {card?.match.map(type => (
            <span key={type} className='mbti-badge mbti-badge--sm'>{type}</span>
          ))}
        </div>
      </div>

      <div className='d-flex gap-2 mt-3'>
        <button className='btn btn-primary' onClick={onRetake}>{t(lang, '重新测试', 'Retake')}</button>
        <Link to='/profile' className='btn btn-outline-secondary'>{t(lang, '去 profile 看看', 'View profile')}</Link>
      </div>
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
  const [lang, setLang]         = useState(() => (navigator.language || '').toLowerCase().startsWith('zh') ? 'zh' : 'en')

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
    const d = await api.get('/api/mbti/questions?mode=' + m)
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
      <div className='mbti-page'>
        <div className='mbti-page__inner'>
          <h1 className='mbti-h'>MBTI 性格测试</h1>
          <p className='text-muted'>{t(lang, '请先登录,好把结果存到你的 profile。', 'Sign in to save your result to your profile.')}</p>
          <Link to='/login' className='btn btn-primary'>{t(lang, '去登录', 'Sign in')}</Link>
        </div>
      </div>
    )
  }

  if (loading) return <div className='mbti-page'><div className='mbti-page__inner text-center py-5'><HandLoader /></div></div>

  return (
    <div className='mbti-page'>
      <div className='mbti-page__inner'>
        <div className='d-flex justify-content-between align-items-center mb-2'>
          <h1 className='mbti-h mbti-h--title'>MBTI<span className='mbti-h__sub'>16 Personalities</span></h1>
          <div className='lang-toggle' data-lang={lang} role='tablist' aria-label={t(lang, '语言', 'Language')}>
            <span className='lang-toggle__thumb' aria-hidden='true' />
            <button type='button' role='tab' aria-selected={lang === 'zh'} className={lang === 'zh' ? 'is-on' : ''} onClick={() => setLang('zh')}>中文</button>
            <button type='button' role='tab' aria-selected={lang === 'en'} className={lang === 'en' ? 'is-on' : ''} onClick={() => setLang('en')}>English</button>
          </div>
        </div>
        {stage === 'intro'  && <Intro existing={existing} onStart={start} lang={lang} />}
        {stage === 'quiz'   && <Quiz questions={questions} mode={mode} onDone={finish} onCancel={() => setStage('intro')} lang={lang} />}
        {stage === 'result' && resultData && <Result data={resultData} onRetake={retake} lang={lang} />}
      </div>
    </div>
  )
}
