import React, { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, useInView } from 'framer-motion';
import {
  ArrowRight,
  Code2,
} from 'lucide-react';
import './LandingPage.css';

/* ────────────────────────────────────────────────────────────────
   FRAMER-MOTION VARIANTS                                          
───────────────────────────────────────────────────────────────── */
const fadeUp = {
  hidden: { opacity: 0, y: 36 },
  visible: (delay: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.85, ease: [0.22, 1, 0.36, 1], delay },
  }),
};

const stagger = {
  hidden:  {},
  visible: { transition: { staggerChildren: 0.11 } },
};

const cardReveal = {
  hidden:  { opacity: 0, y: 48 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: i * 0.13 },
  }),
};

/* ────────────────────────────────────────────────────────────────
   DATA — TECH STACK CARDS (from Stitch page 0)                   
───────────────────────────────────────────────────────────────── */
interface TechStat {
  key: string;
  value: string;
  variant: 'optimized' | 'warning' | 'error' | 'neutral';
}

interface TechCard {
  serial: string;
  title: string;
  body: string;
  stats: TechStat[];
  lamp: 'on' | 'standby' | 'off';
  lampLabel: string;
}

const TECH_CARDS: TechCard[] = [
  {
    serial: 'SYS-01 · RL Engine',
    title: 'Stable-Baselines3',
    body:
      'PPO-driven agents trained on extensive simulations. Adaptive decision making for complex intersection precedence and velocity profiling.',
    stats: [
      { key: 'Status',       value: 'Optimized',  variant: 'optimized' },
      { key: 'Architecture', value: 'PPO-MLP',     variant: 'neutral'   },
    ],
    lamp: 'on',
    lampLabel: 'Live Training',
  },
  {
    serial: 'SYS-02 · Constraint Solver',
    title: 'Google OR-Tools',
    body:
      'Constraint-based scheduling for 100% safety interlocking. Formal verification of all logic paths with zero tolerance for deadlock conditions.',
    stats: [
      { key: 'Logic Verification', value: '0.00%', variant: 'optimized' },
      { key: 'Error Margin',       value: 'None',  variant: 'neutral'   },
    ],
    lamp: 'on',
    lampLabel: 'Verified',
  },
  {
    serial: 'SYS-03 · Visualization',
    title: 'Marey 2.0',
    body:
      'Temporal workspace for human-in-the-loop verification. A complete visual overhaul of classic rail charting for the autonomous era.',
    stats: [
      { key: 'Role',   value: 'Project Architect', variant: 'neutral'  },
    ],
    lamp: 'standby',
    lampLabel: 'Active',
  },
];

/* ────────────────────────────────────────────────────────────────
   DATA — DEVELOPER LINKS                                          
───────────────────────────────────────────────────────────────── */
const DEV_LINKS = [
  { label: 'Source',    icon: Code2,     href: 'https://github.com/swastikjain' },
];

/* ────────────────────────────────────────────────────────────────
   SECTION WRAPPER — uses useInView for entry animation            
───────────────────────────────────────────────────────────────── */
const AnimSection: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = '',
}) => {
  const ref  = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px 0px' });

  return (
    <motion.div
      ref={ref}
      className={className}
      initial="hidden"
      animate={inView ? 'visible' : 'hidden'}
      variants={stagger}
    >
      {children}
    </motion.div>
  );
};

/* ────────────────────────────────────────────────────────────────
   PAGE COMPONENT                                                  
───────────────────────────────────────────────────────────────── */
const Page0: React.FC = () => {
  const navigate = useNavigate();

  /* Update page title */
  useEffect(() => {
    document.title = 'ORBIT | Operational Rail Backbone & Intelligence Tool';
  }, []);

  return (
    <div className="orbit-page">

      {/* ══════════════════════════════════════════════
          NAV — Glassmorphic HUD Header
      ══════════════════════════════════════════════ */}
      <motion.nav
        className="orbit-nav"
        initial={{ opacity: 0, y: -24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.65, ease: [0.22, 1, 0.36, 1] }}
        role="navigation"
        aria-label="Main navigation"
      >
        {/* Logo */}
        <a href="/" className="orbit-nav__logo">
          ORBIT
        </a>

        {/* Nav links */}
        <ul className="orbit-nav__links" role="list">
          <li><a href="#systems"    className="orbit-nav__link">Systems</a></li>
          <li><a href="#simulation" className="orbit-nav__link">Simulation</a></li>
          <li><a href="#network"    className="orbit-nav__link">Network</a></li>
        </ul>

        {/* CTA */}
        <button
          id="nav-launch-btn"
          className="orbit-nav__cta"
          onClick={() => navigate('/dashboard')}
        >
          Launch Mission Control
        </button>
      </motion.nav>

      {/* ══════════════════════════════════════════════
          HERO — Full Viewport
      ══════════════════════════════════════════════ */}
      <header className="orbit-hero" aria-label="Hero">
        {/* Animated CTC grid background */}
        <div className="orbit-hero__grid" aria-hidden="true" />
        {/* Radial glows */}
        <div className="orbit-hero__glow-primary"   aria-hidden="true" />
        <div className="orbit-hero__glow-secondary" aria-hidden="true" />

        <div className="orbit-hero__content">
          {/* "Technological Framework" status badge */}
          <motion.div
            className="orbit-hero__badge"
            custom={0}
            initial="hidden"
            animate="visible"
            variants={fadeUp as any}
            aria-hidden="true"
          >
            <span className="orbit-hero__badge-lamp" />
            Technological Framework — Active
          </motion.div>

          {/* Primary display title */}
          <motion.h1
            className="orbit-hero__title"
            custom={0.1}
            initial="hidden"
            animate="visible"
            variants={fadeUp as any}
          >
            ORBIT
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            className="orbit-hero__subtitle"
            custom={0.2}
            initial="hidden"
            animate="visible"
            variants={fadeUp as any}
          >
            Operational Rail Backbone &amp; Intelligence Tool
          </motion.p>

          {/* Description */}
          <motion.p
            className="orbit-hero__description"
            custom={0.3}
            initial="hidden"
            animate="visible"
            variants={fadeUp as any}
          >
            A unified neural architecture for autonomous train precedence.
            ORBIT synchronizes RL agents with constraint-based optimization
            to eliminate network deadlocks.
          </motion.p>

          {/* Action buttons */}
          <motion.div
            className="orbit-hero__actions"
            custom={0.42}
            initial="hidden"
            animate="visible"
            variants={fadeUp as any}
          >
            <button
              id="hero-launch-btn"
              className="btn-primary"
              onClick={() => navigate('/dashboard')}
            >
              Launch Mission Control <ArrowRight size={13} strokeWidth={2.5} />
            </button>
            <a
              id="hero-architecture-btn"
              href="#intelligence"
              className="btn-secondary"
            >
              View Architecture <ArrowRight size={13} strokeWidth={2.5} />
            </a>
          </motion.div>
        </div>
      </header>

      {/* ══════════════════════════════════════════════
          INTELLIGENCE LAYER — Tech Stack
      ══════════════════════════════════════════════ */}
      <section
        id="intelligence"
        className="orbit-section orbit-section--surface"
        aria-labelledby="intelligence-heading"
      >
        <AnimSection>
          {/* Section header */}
          <motion.p className="orbit-section__eyebrow" variants={fadeUp as any}>
            Core Architecture
          </motion.p>
          <motion.h2
            id="intelligence-heading"
            className="orbit-section__title"
            variants={fadeUp as any}
          >
            Intelligence Layer
          </motion.h2>
          <motion.p className="orbit-section__subtitle" variants={fadeUp as any}>
            Merging Reinforcement Learning with formal methods for
            high-fidelity rail operations.
          </motion.p>
        </AnimSection>

        {/* 3-column machined cards */}
        <div className="orbit-tech-grid" id="systems">
          {TECH_CARDS.map((card, i) => (
            <motion.article
              key={card.title}
              className="orbit-card"
              custom={i}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: '-60px' }}
              variants={cardReveal as any}
              aria-label={card.title}
            >
              {/* Serial / Eyebrow */}
              <p className="orbit-card__serial">{card.serial}</p>

              {/* Title — Gauge Label */}
              <h3 className="orbit-card__title">{card.title}</h3>

              {/* Description */}
              <p className="orbit-card__body">{card.body}</p>

              {/* Tonal separator (no 1px border) */}
              <div className="orbit-card__separator" aria-hidden="true" />

              {/* Telemetry stats */}
              <div className="orbit-card__stats">
                {card.stats.map(stat => (
                  <div key={stat.key} className="orbit-card__stat-row">
                    <span className="orbit-card__stat-key">{stat.key}</span>
                    <span className={`orbit-card__stat-val orbit-card__stat-val--${stat.variant}`}>
                      {stat.value}
                    </span>
                  </div>
                ))}

                {/* Signal Lamp indicator */}
                <div className="orbit-card__stat-row" style={{ marginTop: 6 }}>
                  <span className="orbit-lamp">
                    <span className={`orbit-lamp__dot orbit-lamp__dot--${card.lamp}`} />
                    <span className="orbit-lamp__label">{card.lampLabel}</span>
                  </span>
                </div>
              </div>
            </motion.article>
          ))}
        </div>
      </section>

      {/* ══════════════════════════════════════════════
          DEVELOPER / ARCHITECT SECTION
      ══════════════════════════════════════════════ */}
      <section
        className="orbit-developer"
        id="network"
        aria-labelledby="developer-heading"
      >
        <AnimSection className="orbit-developer__inner">
          {/* Quote */}
          <motion.blockquote
            className="orbit-developer__quote"
            variants={fadeUp as any}
            cite="Swastik Jain"
          >
            "Engineering efficient systems for complex infrastructure."
          </motion.blockquote>

          {/* Name */}
          <motion.p
            id="developer-heading"
            className="orbit-developer__name"
            variants={fadeUp as any}
          >
            Developed by Swastik
          </motion.p>

          {/* Affiliation */}
          <motion.p className="orbit-developer__affiliation" variants={fadeUp as any}>
            Madhav Institute of Technology and Science
          </motion.p>

          {/* Link grid — tactile secondary buttons */}
          <motion.div className="orbit-developer__links" variants={fadeUp as any} role="list">
            {DEV_LINKS.map((link) => (
              <a
                key={link.label}
                href={link.href}
                className="orbit-dev-link"
                target={link.href.startsWith('http') ? '_blank' : undefined}
                rel={link.href.startsWith('http') ? 'noopener noreferrer' : undefined}
                role="listitem"
                id={`dev-link-${link.label.toLowerCase()}`}
              >
                <link.icon size={11} strokeWidth={2} />
                {link.label}
              </a>
            ))}
          </motion.div>
        </AnimSection>
      </section>


      {/* ══════════════════════════════════════════════
          FOOTER
      ══════════════════════════════════════════════ */}
      <footer className="orbit-footer pb-12" role="contentinfo">
        <p className="orbit-footer__copy">
          © 2024 ORBIT Systems. Built for Precision.
        </p>
        <p className="orbit-footer__version">
          v2.0.0 — AI Train Precedence
        </p>
      </footer>

    </div>
  );
};

export default Page0;
