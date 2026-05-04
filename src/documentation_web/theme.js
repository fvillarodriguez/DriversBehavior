/* =============================================================
   Editorial Lab — runtime
   - Reveal motion (IntersectionObserver)
   - Equation inspector (hover/tap popovers)
   - Auto-height postMessage to Streamlit parent
   - Smooth-scroll TOC
   ============================================================= */

(function () {
  'use strict';

  /* ------------ Reveal motion --------------------------------- */
  const revealEls = document.querySelectorAll('.reveal');
  if (revealEls.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('in');
            io.unobserve(e.target);
          }
        });
      },
      { rootMargin: '0px 0px -8% 0px', threshold: 0.05 }
    );
    revealEls.forEach((el) => io.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add('in'));
  }

  /* ------------ Equation inspector ---------------------------- */
  // Each .lab-eq.has-inspector has a sibling .eq-inspector (or one nested as child)
  // Hover desktop / tap mobile to open.
  const eqs = document.querySelectorAll('.lab-eq.has-inspector');
  let openInspector = null;
  function closeInspector() {
    if (openInspector) {
      const openEq = openInspector.closest('.lab-eq.has-inspector');
      openInspector.removeAttribute('data-open');
      if (openEq) openEq.classList.remove('is-inspector-open');
      openInspector = null;
    }
  }
  eqs.forEach((eq) => {
    const ins = eq.querySelector(':scope > .eq-inspector');
    if (!ins) return;
    let hoverTimer;
    const open = () => {
      closeInspector();
      ins.setAttribute('data-open', 'true');
      eq.classList.add('is-inspector-open');
      openInspector = ins;
    };
    const close = () => {
      ins.removeAttribute('data-open');
      eq.classList.remove('is-inspector-open');
      if (openInspector === ins) openInspector = null;
    };
    // Desktop: hover w/ small delay
    eq.addEventListener('mouseenter', () => {
      clearTimeout(hoverTimer);
      hoverTimer = setTimeout(open, 140);
    });
    eq.addEventListener('mouseleave', () => {
      clearTimeout(hoverTimer);
      hoverTimer = setTimeout(close, 220);
    });
    ins.addEventListener('mouseenter', () => clearTimeout(hoverTimer));
    ins.addEventListener('mouseleave', close);
    // Touch / click toggle
    eq.addEventListener('click', (ev) => {
      if (ev.target.closest('.eq-inspector')) return;
      if (ins.getAttribute('data-open') === 'true') close();
      else open();
      ev.stopPropagation();
    });
  });
  document.addEventListener('click', (ev) => {
    if (!ev.target.closest('.lab-eq')) closeInspector();
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') closeInspector();
  });

  /* ------------ Hero TOC navigation --------------------------- */
  const docNavConfig = window.LAB_DOC_NAV || {};
  const docNavSections = docNavConfig.sections || {};

  function normalizeText(value) {
    return (value || '').replace(/\s+/g, ' ').trim();
  }

  function currentDocLang() {
    return (document.documentElement.getAttribute('lang') || 'es')
      .slice(0, 2)
      .toLowerCase();
  }

  function scrollToAnchor(hash) {
    if (!hash || hash.charAt(0) !== '#') return false;
    const target = document.getElementById(hash.slice(1));
    if (!target) return false;
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return true;
  }

  function matchesRadioLabel(el, expected) {
    const aria = normalizeText(el.getAttribute('aria-label'));
    const text = normalizeText(el.textContent);
    return aria === expected || text === expected;
  }

  function clickParentStreamlitSection(sectionId) {
    const section = docNavSections[sectionId];
    if (!section || !section.labels) return false;

    const lang = currentDocLang();
    const expected = section.labels[lang] || section.labels.es || section.labels.en;
    if (!expected) return false;

    let parentDoc;
    try {
      if (!window.parent || window.parent === window) return false;
      parentDoc = window.parent.document;
    } catch (_) {
      return false;
    }

    const candidates = parentDoc.querySelectorAll(
      'section[data-testid="stSidebar"] [role="radio"], ' +
      'section[data-testid="stSidebar"] label'
    );
    for (const el of candidates) {
      if (!matchesRadioLabel(el, expected)) continue;
      if (
        el.getAttribute('aria-checked') === 'true' ||
        el.querySelector('input:checked')
      ) {
        return true;
      }
      el.click();
      return true;
    }
    return false;
  }

  document.querySelectorAll('.hero-toc a[data-section]').forEach((a) => {
    a.addEventListener('click', (ev) => {
      ev.preventDefault();
      const sectionId = a.getAttribute('data-section');
      if (clickParentStreamlitSection(sectionId)) return;
      scrollToAnchor(a.getAttribute('href'));
    });
  });

  /* ------------ Smooth-scroll for in-page anchors ------------- */
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    if (a.matches('.hero-toc a[data-section]')) return;
    a.addEventListener('click', (ev) => {
      const id = a.getAttribute('href').slice(1);
      const target = document.getElementById(id);
      ev.preventDefault();
      if (!target) return;
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  /* ------------ Auto-height bridge (Streamlit iframe) --------- */
  // Streamlit's components.v1.html sandboxes us in an iframe with a fixed height.
  // We can't change Streamlit's render call from inside, but we CAN size the
  // body so that the host iframe scrollbar disappears when the host height
  // is set generously. As a best-effort, we also broadcast a setFrameHeight
  // message that custom hosts may pick up, and we observe content size.
  function reportHeight() {
    try {
      const h = Math.max(
        document.documentElement.scrollHeight,
        document.body.scrollHeight
      );
      if (window.parent && window.parent !== window) {
        window.parent.postMessage(
          { type: 'lab:setFrameHeight', height: h },
          '*'
        );
        window.parent.postMessage(
          {
            type: 'streamlit:setFrameHeight',
            height: h,
          },
          '*'
        );
      }
    } catch (_) { /* cross-origin, ignore */ }
  }
  if ('ResizeObserver' in window) {
    const ro = new ResizeObserver(reportHeight);
    ro.observe(document.body);
    ro.observe(document.documentElement);
  }
  window.addEventListener('load', reportHeight);
  window.addEventListener('resize', reportHeight);
  // Re-report after MathJax typesets (fonts load shifts heights)
  if (window.MathJax) {
    if (window.MathJax.startup && window.MathJax.startup.promise) {
      window.MathJax.startup.promise.then(reportHeight);
    }
    window.addEventListener('mathjax-typeset', reportHeight);
  }

  /* ------------ Theme bootstrap (system preference) ----------- */
  // The host already sets [data-theme] on <html>; here we add a class on body
  // when system preference disagrees, for any future overrides.
  const root = document.documentElement;
  const explicit = root.getAttribute('data-theme');
  if (!explicit && window.matchMedia) {
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.setAttribute('data-theme', dark ? 'dark' : 'light');
  }

  /* ------------ Search highlight & scroll-to-first ------------- */
  function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  // Skip these element names entirely — they contain non-prose content
  // (rendered MathJax, code/style/script blocks, already-marked ranges).
  const SEARCH_SKIP_TAGS = new Set([
    'SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA',
    'MARK', 'MJX-CONTAINER', 'MJX-MATH', 'MJX-MO', 'MJX-MI', 'MJX-MN',
    'MJX-MROW', 'MJX-MFRAC', 'MJX-MSUB', 'MJX-MSUP', 'MJX-MSUBSUP',
  ]);

  function highlightSearch(query) {
    if (!query || query.length < 2) return 0;
    const root = document.querySelector('.lab-main') || document.body;
    if (!root) return 0;
    const re = new RegExp(escapeRegExp(query), 'gi');

    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (!node.nodeValue || !node.nodeValue.trim()) {
            return NodeFilter.FILTER_REJECT;
          }
          // Walk up the parent chain; reject if any ancestor is in skip list,
          // or carries .formula (raw LaTeX) or .num (eq number).
          let p = node.parentNode;
          while (p && p !== root) {
            if (SEARCH_SKIP_TAGS.has(p.nodeName)) {
              return NodeFilter.FILTER_REJECT;
            }
            if (p.classList && (
              p.classList.contains('formula') ||
              p.classList.contains('hero-marker')
            )) {
              return NodeFilter.FILTER_REJECT;
            }
            p = p.parentNode;
          }
          re.lastIndex = 0;
          return re.test(node.nodeValue)
            ? NodeFilter.FILTER_ACCEPT
            : NodeFilter.FILTER_REJECT;
        },
      }
    );

    const targets = [];
    while (walker.nextNode()) targets.push(walker.currentNode);

    let total = 0;
    targets.forEach((textNode) => {
      const original = textNode.nodeValue;
      re.lastIndex = 0;
      const replaced = original.replace(re, (m) => {
        total += 1;
        return `<mark class="lab-mark">${escapeHtml(m)}</mark>`;
      });
      const span = document.createElement('span');
      span.innerHTML = replaced;
      // Replace the text node with the constructed span (preserves marks).
      textNode.parentNode.replaceChild(span, textNode);
    });
    return total;
  }

  function applySearchFromMeta() {
    const meta = document.querySelector('meta[name="lab-query"]');
    const q = meta ? (meta.getAttribute('content') || '').trim() : '';
    if (!q) return;
    const hits = highlightSearch(q);
    if (hits > 0) {
      const first = document.querySelector('mark.lab-mark');
      if (first) {
        // Defer to next frame so layout has settled.
        requestAnimationFrame(() => {
          first.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
      }
    }
  }

  // Run highlighting after MathJax finishes typesetting, otherwise we may
  // mangle text inside half-rendered <mjx-*> nodes. If MathJax is absent,
  // run on next animation frame.
  if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(applySearchFromMeta);
  } else {
    requestAnimationFrame(applySearchFromMeta);
  }
})();
