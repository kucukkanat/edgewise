// Edgewise docs: navigation, search, copy buttons, table of contents and the models table.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

/* copy buttons */
function copy(btn, text) {
  const done = (ok) => {
    const o = btn.textContent;
    btn.textContent = ok ? 'Copied' : 'Press Ctrl+C';
    setTimeout(() => (btn.textContent = o), 1400);
  };
  navigator.clipboard?.writeText(text).then(() => done(true), () => done(false)) ?? done(false);
}
for (const b of $$('figure.code .copy')) b.addEventListener('click', () => copy(b, b.closest('figure').querySelector('code').textContent));
for (const b of $$('[data-copy]')) b.addEventListener('click', () => copy(b, b.dataset.copy));

/* mobile nav */
const menuBtn = $('#menuBtn');
const scrim = $('#scrim');
const closeNav = () => {
  document.body.classList.remove('nav-open');
  scrim.hidden = true;
  menuBtn.setAttribute('aria-expanded', 'false');
};
menuBtn.addEventListener('click', () => {
  const open = !document.body.classList.contains('nav-open');
  document.body.classList.toggle('nav-open', open);
  scrim.hidden = !open;
  menuBtn.setAttribute('aria-expanded', String(open));
});
scrim.addEventListener('click', closeNav);

/* theme */
$('#themeBtn')?.addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  const next = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem('ew-theme', next);
  } catch {}
});

/* table of contents */
const heads = $$('.page h2[id]');
if (heads.length && 'IntersectionObserver' in window) {
  const obs = new IntersectionObserver(
    (es) => {
      for (const e of es) if (e.isIntersecting) for (const a of $$('#tocList a')) a.classList.toggle('on', a.getAttribute('href') === `#${e.target.id}`);
    },
    { rootMargin: '-70px 0px -70% 0px' },
  );
  for (const h of heads) obs.observe(h);
}

/* search */
const q = $('#q');
const results = $('#results');
let index = null;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
async function search(term) {
  index ??= await fetch('assets/search.json').then((r) => r.json());
  const t = term.toLowerCase().trim();
  if (!t) return [];
  const words = t.split(/\s+/);
  const hits = [];
  for (const p of index) {
    const hay = `${p.title} ${p.text}`.toLowerCase();
    if (!words.every((w) => hay.includes(w))) continue;
    let score = p.title.toLowerCase().includes(t) ? 10 : 0;
    const h = p.headings.find((x) => x.text.toLowerCase().includes(words[0]));
    if (h) score += 5;
    const i = hay.indexOf(words[0]);
    const snippet = p.text.slice(Math.max(0, i - p.title.length - 40), Math.max(0, i - p.title.length - 40) + 120);
    hits.push({ p, h, score, snippet });
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, 8);
}
let sel = -1;
q.addEventListener('input', async () => {
  const hits = await search(q.value);
  sel = -1;
  results.hidden = !q.value.trim();
  results.innerHTML = hits.length
    ? hits.map((x) => `<a href="${x.p.slug}.html${x.h ? `#${x.h.id}` : ''}"><b>${esc(x.p.title)}</b>${x.h ? ` <span>› ${esc(x.h.text)}</span>` : ''}<small>${esc(x.snippet)}…</small></a>`).join('')
    : '<p class="none">No matches</p>';
});
q.addEventListener('keydown', (e) => {
  const links = $$('a', results);
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    sel = Math.max(0, Math.min(links.length - 1, sel + (e.key === 'ArrowDown' ? 1 : -1)));
    links.forEach((a, i) => a.classList.toggle('sel', i === sel));
  } else if (e.key === 'Enter' && links.length) location.href = links[Math.max(0, sel)].href;
  else if (e.key === 'Escape') results.hidden = true;
});
document.addEventListener('click', (e) => {
  if (!e.target.closest('.search')) results.hidden = true;
});
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement !== q && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) {
    e.preventDefault();
    q.focus();
  }
});

/* models table filters */
const table = $('#mtable');
if (table) {
  let verb = 'all';
  let input = 'any';
  const seg = (el, opts, get, set) => {
    for (const [v, l] of opts) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = l;
      b.setAttribute('aria-pressed', String(v === get()));
      b.addEventListener('click', () => {
        set(v);
        for (const x of $$('button', el)) x.setAttribute('aria-pressed', String(x === b));
        render();
      });
      el.append(b);
    }
  };
  seg($('#verbSeg'), [['all', 'All verbs'], ...['generate', 'evaluate', 'embed', 'speak', 'paint', 'forecast', 'vad'].map((v) => [v, v])], () => verb, (v) => (verb = v));
  seg($('#inSeg'), [['any', 'Any input'], ...['text', 'image', 'audio', 'series'].map((v) => [v, v])], () => input, (v) => (input = v));
  $('#stableOnly').addEventListener('change', render);
  let webgpu = false;
  function render() {
    const stable = $('#stableOnly').checked;
    let n = 0;
    let runnable = 0;
    for (const tr of $$('tbody tr', table)) {
      const show = (verb === 'all' || tr.dataset.verb === verb) && (input === 'any' || tr.dataset.accepts.split(' ').includes(input)) && (!stable || tr.dataset.status === 'stable');
      tr.hidden = !show;
      const needsGpu = !/WASM|CPU/.test(tr.children[4].textContent);
      tr.classList.toggle('unrunnable', needsGpu && !webgpu);
      if (show) {
        n++;
        if (!(needsGpu && !webgpu)) runnable++;
      }
    }
    $('#mcount').textContent = `${n} models shown · ${runnable} can run in this browser`;
  }
  render();
  (async () => {
    try {
      webgpu = !!(await navigator.gpu?.requestAdapter());
    } catch {}
    render();
  })();
}
