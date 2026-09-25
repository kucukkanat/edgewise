// Live demos and the capability probe. These run the real Edgewise build in lib/.
const lib = import('../lib/index.js').then((m) => {
  const ew = m.default;
  // ?hub=http://host lets you point the demos at a mirror (used for testing the site offline).
  const hub = new URLSearchParams(location.search).get('hub');
  if (hub) ew.configure({ hub: `${hub}/hf`, wasmPaths: `${hub}/cdn/npm/onnxruntime-web@${document.querySelector('meta[name="ort-web"]')?.content}/dist/` });
  return ew;
});
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);

/* capability probe (intro page) */
const probe = $('#probeOut');
if (probe) {
  (async () => {
    try {
      const { capabilities, registry } = await lib;
      const c = await capabilities();
      const fmt = (v) => (v === null ? '<span class="t">null</span>' : typeof v === 'boolean' ? `<span class="t">${v}</span>` : typeof v === 'number' ? `<span class="f">${v}</span>` : `<span class="s">'${esc(v)}'</span>`);
      const rows = { runtime: c.runtime, webgpu: c.webgpu, shaderF16: c.shaderF16, adapter: c.adapter, hardwareGpu: c.hardwareGpu, crossOriginIsolated: c.crossOriginIsolated, threads: c.threads, builtinAI: c.builtinAI, cores: c.cores, tier: c.tier };
      probe.innerHTML = `{\n${Object.entries(rows).map(([k, v]) => `  <span class="k">${k}</span>: ${fmt(v)},`).join('\n')}\n}`;
      const all = registry.list();
      const ok = registry.list({ runnable: true });
      const verbs = new Set(ok.map((m) => m.verb));
      $('#probeFoot').innerHTML = `This browser can run <b>${ok.length} of ${all.length}</b> models, covering <b>${[...verbs].filter((v) => v !== 'vad').length} of 6</b> verbs${c.webgpu ? ', with WebGPU.' : '. Without WebGPU, models run on WebAssembly and <code>paint</code> is unavailable.'} <a href="models.html" style="color:var(--tk-kw)">See which</a>.`;
    } catch (e) {
      probe.textContent = `// could not load Edgewise: ${e.message}`;
    }
  })();
}

/* demos */
function status(btn, text) {
  btn.parentElement.querySelector('.demo-status').textContent = text;
}
function progress(btn) {
  const files = new Map();
  return (e) => {
    if (e.type === 'download') {
      files.set(e.file, e);
      let l = 0;
      let t = 0;
      for (const f of files.values()) {
        l += f.loaded;
        t += f.total;
      }
      status(btn, `downloading ${Math.round((l / (t || 1)) * 100)}% of ${Math.round(t / 1e6)} MB`);
    } else if (e.type === 'compile') status(btn, 'compiling…');
    else if (e.type === 'ready') status(btn, `running on ${e.device}…`);
  };
}
const done = (btn, info, t0) => status(btn, `${info.model} · ${info.device} · ${info.dtype} · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
function bars(el, rows) {
  el.innerHTML = rows.map(([label, v]) => `<div class="bar"><span>${esc(label)}</span><span class="t"><i style="width:${Math.max(1, v * 100)}%"></i></span><b>${v.toFixed(2)}</b></div>`).join('');
}

const demos = {
  async generate(btn) {
    const { generate } = await lib;
    const out = $('#gen-out');
    out.textContent = '';
    const t0 = performance.now();
    const run = generate({ model: 'text:tiny', input: $('#gen-in').value, maxTokens: 160, onProgress: progress(btn) });
    for await (const d of run) out.textContent += d;
    done(btn, (await run).info, t0);
  },
  async evaluate(btn) {
    const { evaluate, choice } = await lib;
    const t0 = performance.now();
    const lanes = { billing: 'billing and payments', tech: 'technical support', sales: 'buying or pricing', chat: 'small talk' };
    const r = await evaluate({ model: 'judge:router', state: $('#ev-in').value, questions: { lane: choice(lanes) }, onProgress: progress(btn) });
    bars($('#ev-out'), Object.entries(r.answers.lane.probabilities).sort((a, b) => b[1] - a[1]));
    done(btn, r.info.lane, t0);
  },
  async embed(btn) {
    const { embed } = await lib;
    const lines = $('#em-in').value.split('\n').map((s) => s.trim()).filter(Boolean);
    const t0 = performance.now();
    const { embeddings, info } = await embed({ model: 'embed:tiny', values: lines, onProgress: progress(btn) });
    const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
    bars($('#em-out'), lines.slice(1).map((l, i) => [l, Math.max(0, dot(embeddings[0], embeddings[i + 1]))]));
    done(btn, info, t0);
  },
  async speak(btn) {
    const { speak } = await lib;
    const t0 = performance.now();
    const audio = await speak({ model: 'voice:default', voice: $('#sp-voice').value, input: $('#sp-in').value, onProgress: progress(btn) }).play();
    done(btn, audio.info, t0);
  },
  async transcribe(btn) {
    const { generate, mic } = await lib;
    const m = await mic();
    await m.start();
    status(btn, 'recording… release to stop');
    await new Promise((r) => btn.addEventListener('pointerup', r, { once: true }));
    const samples = await m.stop();
    m.dispose();
    const t0 = performance.now();
    const r = await generate({ model: 'stt:tiny', input: samples, onProgress: progress(btn) });
    $('#tr-out').textContent = r.text || '(no speech heard)';
    done(btn, r.info, t0);
  },
  async forecast(btn) {
    const { forecast } = await lib;
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const base = (t) => 48 + 22 * Math.sin(((t - 6) / 24) * 2 * Math.PI) + t * 0.08;
    const H = 96;
    const F = 24;
    const hist = Array.from({ length: H }, (_, t) => base(t) + (rnd() - 0.5) * 9);
    const t0 = performance.now();
    const r = await forecast({ model: 'forecast:default', series: hist, horizon: F, onProgress: progress(btn) });
    chart($('#fc-out'), hist, Array.from(r.median), Array.from(r.quantiles[0.1]), Array.from(r.quantiles[0.9]));
    done(btn, r.info, t0);
  },
  async paint(btn) {
    const { paint, capabilities } = await lib;
    if (!(await capabilities()).webgpu) return status(btn, 'this browser has no WebGPU, so paint cannot run here');
    const cv = $('#pt-out');
    const ctx = cv.getContext('2d');
    const t0 = performance.now();
    const run = paint({ model: 'image:default', prompt: $('#pt-in').value, size: '512x512', steps: 1, allowPreview: true, onProgress: progress(btn) });
    for await (const s of run) {
      const tmp = new OffscreenCanvas(s.preview.width, s.preview.height);
      tmp.getContext('2d').putImageData(new ImageData(s.preview.data, s.preview.width, s.preview.height), 0, 0);
      ctx.drawImage(tmp, 0, 0, 512, 512);
    }
    const r = await run;
    ctx.putImageData(r.image.toImageData(), 0, 0);
    done(btn, r.info, t0);
  },
};

function chart(svg, hist, mid, lo, hi) {
  const H = hist.length;
  const F = mid.length;
  const W = 640;
  const Ht = 230;
  const L = 40;
  const R = 12;
  const T = 12;
  const B = 28;
  const all = [...hist, ...lo, ...hi];
  const yMin = Math.floor(Math.min(...all) / 10) * 10;
  const yMax = Math.ceil(Math.max(...all) / 10) * 10;
  const x = (i) => L + (i / (H + F - 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - yMin) / (yMax - yMin)) * (Ht - T - B);
  const path = (pts) => `M${pts.map(([a, b]) => `${x(a).toFixed(1)} ${y(b).toFixed(1)}`).join(' L')}`;
  const band = `${path(hi.map((v, i) => [H + i, v]))} L${lo.map((v, i) => [H + F - 1 - i, lo[F - 1 - i]]).map(([a, b]) => `${x(a).toFixed(1)} ${y(b).toFixed(1)}`).join(' L')} Z`;
  svg.innerHTML = `
    ${[0, 0.25, 0.5, 0.75, 1].map((f) => { const v = yMin + f * (yMax - yMin); return `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" style="stroke:var(--line)"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end" style="fill:var(--muted);font:11px var(--f-mono)">${Math.round(v)}</text>`; }).join('')}
    <path d="${band}" style="fill:color-mix(in srgb,var(--accent) 22%,transparent)"/>
    <line x1="${x(H - 0.5)}" x2="${x(H - 0.5)}" y1="${T}" y2="${Ht - B}" style="stroke:var(--line-2)" stroke-dasharray="3 4"/>
    <path d="${path(hist.map((v, i) => [i, v]))}" style="fill:none;stroke:var(--ink-2)" stroke-width="1.6"/>
    <path d="${path([[H - 1, hist[H - 1]], ...mid.map((v, i) => [H + i, v])])}" style="fill:none;stroke:var(--accent)" stroke-width="2"/>
    <text x="${x(H - 0.5)}" y="${Ht - 8}" text-anchor="middle" style="fill:var(--muted);font:11px var(--f-mono)">now</text>`;
}

for (const btn of document.querySelectorAll('[data-run]')) {
  const name = btn.dataset.run;
  const start = async () => {
    if (btn.disabled) return;
    btn.disabled = true;
    try {
      await demos[name](btn);
    } catch (e) {
      status(btn, `error: ${e.message}${e.hint ? ` · ${e.hint}` : ''}`);
      console.error(e);
    } finally {
      btn.disabled = false;
    }
  };
  if (name === 'transcribe') btn.addEventListener('pointerdown', start);
  else btn.addEventListener('click', start);
}
