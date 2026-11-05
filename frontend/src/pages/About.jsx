import React from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../App'

/* Content lives in plain arrays so the layout below stays dumb. When this
   page opens up to every member, these blocks move into user data without
   the markup having to change. */
const PROFILE = {
  username: 'horizon',
  display_name: 'Horizon',
  avatar_url: null,
  tagline: '一个在圣路易斯写代码、做社区的人',
  location: 'St. Louis, MO',
  email: 'your@email.com',
  now: [
    '正在写 Arch Bay —— 一个给圣路易斯华人用的轻量社区站。',
    '白天写代码,晚上看看邮件、改改小功能,顺手记一条 Memo。',
    '想把站子做小而稳,不急着长大。',
  ],
  timeline: [
    { year: '2026', text: 'Arch Bay 上线,邀请制开放给朋友试用。' },
    { year: '2025', text: '开始在圣路易斯做本地社区相关的产品尝试。' },
    { year: '2024', text: '转到更偏产品的方向,边做边学。' },
    { year: '2023', text: '从零开始写第一版后端,练手全栈。' },
  ],
  projects: [
    {
      title: 'Arch Bay',
      body: '圣路易斯华人的日常小站: Market、转租、留言板、Tarot、八字、Memo。想做一个不打扰、留得下、用得顺手的地方。',
    },
    {
      title: '未来计划',
      body: '把个人主页放开给每位成员,让大家写几句自我介绍 —— 认识彼此比刷信息流更要紧。再慢慢加上活动、兴趣小组这些。',
    },
  ],
}

function Hero({ me }) {
  const initial = (me.display_name || me.username || '?').trim().charAt(0).toUpperCase()
  return (
    <section className="about-hero">
      <div className="about-hero__avatar" aria-hidden={me.avatar_url ? 'false' : 'true'}>
        {me.avatar_url
          ? <img src={me.avatar_url} alt="" />
          : <span>{initial}</span>}
      </div>
      <div className="about-hero__meta">
        <h1 className="about-hero__name">{me.display_name}</h1>
        <p className="about-hero__tagline">{me.tagline}</p>
        <p className="about-hero__location">
          <i className="fas fa-location-dot" aria-hidden="true" />
          <span>{me.location}</span>
        </p>
      </div>
    </section>
  )
}

function Section({ eyebrow, title, children }) {
  return (
    <section className="about-section">
      <div className="about-section__head">
        <span className="about-section__eyebrow">{eyebrow}</span>
        <h2 className="about-section__title">{title}</h2>
      </div>
      {children}
    </section>
  )
}

function NowList({ items }) {
  return (
    <ul className="about-now">
      {items.map((line, i) => (
        <li key={i} className="about-now__item">{line}</li>
      ))}
    </ul>
  )
}

function Timeline({ items }) {
  return (
    <ol className="about-timeline">
      {items.map(({ year, text }, i) => (
        <li key={i} className="about-timeline__row">
          <span className="about-timeline__year">{year}</span>
          <span className="about-timeline__text">{text}</span>
        </li>
      ))}
    </ol>
  )
}

function Projects({ items }) {
  return (
    <div className="about-projects">
      {items.map(({ title, body }, i) => (
        <article key={i} className="about-projects__card">
          <h3 className="about-projects__title">{title}</h3>
          <p className="about-projects__body">{body}</p>
        </article>
      ))}
    </div>
  )
}

function Contact({ me, onMessage }) {
  return (
    <div className="about-contact">
      <button type="button" className="about-contact__msg" onClick={onMessage}>
        <i className="fas fa-paper-plane" aria-hidden="true" />
        <span>站内私信</span>
      </button>
      <a className="about-contact__mail" href={`mailto:${me.email}`}>
        <i className="fas fa-envelope" aria-hidden="true" />
        <span>{me.email}</span>
      </a>
    </div>
  )
}

/* The route is registered for everyone so a pasted link resolves, but the
   page stays private until it is ready: only the owner (username === horizon)
   or an admin sees content. Everyone else gets a gentle empty state. */
export default function About() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const isOwner  = user?.username === 'horizon'
  const isAdmin  = user?.role_info?.permissions?.includes('admin')
  const canView  = isOwner || isAdmin

  if (!canView) {
    return (
      <div className="about-empty">
        <h1 className="about-empty__title">该页面尚未公开</h1>
        <p className="about-empty__body">
          个人主页还在搭,之后会开放给每位成员。先回 <Link to="/home">首页</Link> 看看。
        </p>
      </div>
    )
  }

  const openMessage = () => {
    navigate('/friends', {
      state: {
        openChat: {
          username: PROFILE.username,
          display_name: PROFILE.display_name,
        },
      },
    })
  }

  return (
    <div className="about-page">
      <Hero me={PROFILE} />

      <Section eyebrow="Now" title="现在在做什么">
        <NowList items={PROFILE.now} />
      </Section>

      <Section eyebrow="Timeline" title="做过的事">
        <Timeline items={PROFILE.timeline} />
      </Section>

      <Section eyebrow="Work" title="我在做的事">
        <Projects items={PROFILE.projects} />
      </Section>

      <Section eyebrow="Contact" title="联系">
        <Contact me={PROFILE} onMessage={openMessage} />
      </Section>
    </div>
  )
}
