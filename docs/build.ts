/**
 * Builds the documentation site into docs/dist.
 *
 *   bun --conditions=source docs/build.ts        # pages, models table, search index, demo bundle
 *   bun --conditions=source docs/build.ts --api  # also runs TypeDoc into docs/dist/api
 *
 * Pages are Markdown files in docs/content with a small front-matter block:
 *   ---
 *   title: generate
 *   group: The six verbs
 *   order: 10
 *   eyebrow: verb · → text or object
 *   ---
 */
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { Marked, type Tokens } from 'marked';
import Prism from 'prismjs';
import 'prismjs/components/prism-typescript.js';
import 'prismjs/components/prism-jsx.js';
import 'prismjs/components/prism-tsx.js';
import 'prismjs/components/prism-bash.js';
import 'prismjs/components/prism-json.js';
import pkg from '../package.json' with { type: 'json' };
import { registry } from '../src/index.ts';

const root = new URL('.', import.meta.url).pathname;
const contentDir = join(root, 'content');
const out = join(root, 'dist');
const base = process.env.DOCS_BASE ?? '/edgewise/';
const withApi = process.argv.includes('--api');

interface Page {
  slug: string;
  title: string;
  group: string;
  order: number;
  eyebrow?: string;
  description?: string;
  body: string;
  headings: { id: string; text: string }[];
  html: string;
  text: string;
}

const GROUPS = ['Get started', 'The six verbs', 'Inputs', 'Guides', 'Extras', 'Reference'];

function frontMatter(src: string): { meta: Record<string, string>; body: string } {
  const m = /^---\n([\s\S]*?)\n---\n?/.exec(src);
  if (!m) return { meta: {}, body: src };
  const meta: Record<string, string> = {};
  for (const line of m[1].split('\n')) {
    const i = line.indexOf(':');
    if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return { meta, body: src.slice(m[0].length) };
}

const slugify = (s: string) =>
  s
    .toLowerCase()
    .replace(/<[^>]+>/g, '')
    .replace(/[`'"().,:;!?/&]+/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');

const decode = (s: string) =>
  s
    .replace(/&#39;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

function highlight(code: string, lang: string): string {
  const grammar = Prism.languages[lang === 'sh' || lang === 'shell' ? 'bash' : lang];
  return grammar ? Prism.highlight(code, grammar, lang) : esc(code);
}

function renderPage(slug: string, body: string, pageSlugs: Set<string>): { html: string; headings: { id: string; text: string }[] } {
  const headings: { id: string; text: string }[] = [];
  const used = new Set<string>();
  const marked = new Marked({ gfm: true });
  marked.use({
    renderer: {
      heading({ tokens, depth }: Tokens.Heading) {
        const inner = this.parser.parseInline(tokens);
        if (depth === 1) return `<h1>${inner}</h1>\n`;
        let id = slugify(inner);
        while (used.has(id)) id += '-';
        used.add(id);
        if (depth === 2) headings.push({ id, text: decode(inner.replace(/<[^>]+>/g, '')) });
        return `<h${depth} id="${id}">${inner}<a class="anchor" href="#${id}" aria-label="Link to section">#</a></h${depth}>\n`;
      },
      code({ text, lang }: Tokens.Code) {
        const [language = 'ts', ...rest] = (lang ?? 'ts').split(/\s+/);
        const file = rest.find((r) => r.startsWith('file='))?.slice(5) ?? '';
        return `<figure class="code"><div class="code-head"><span class="file">${esc(file)}</span><span class="lang">${esc(language)}</span><button class="copy" type="button">Copy</button></div><pre class="language-${esc(language)}"><code class="language-${esc(language)}">${highlight(text, language)}</code></pre></figure>\n`;
      },
      blockquote({ tokens }: Tokens.Blockquote) {
        const inner = this.parser.parse(tokens);
        const m = /^<p>\[!(TIP|NOTE|WARN|RISK)\]\s*/.exec(inner);
        if (!m) return `<blockquote>${inner}</blockquote>\n`;
        const kind = m[1].toLowerCase() === 'note' ? 'tip' : m[1].toLowerCase();
        const label = { tip: 'Tip', warn: 'Note', risk: 'Careful' }[kind] ?? 'Note';
        return `<div class="note ${kind}"><span class="tag">${label}</span><div><p>${inner.slice(m[0].length)}</div></div>\n`;
      },
      link({ href, title, tokens }: Tokens.Link) {
        const text = this.parser.parseInline(tokens);
        let h = href;
        // `#page` or `page#section` links between pages become relative .html links.
        const m = /^#?([a-z0-9-]+)(#[a-z0-9-]+)?$/.exec(href);
        if (m && pageSlugs.has(m[1]) && (href.startsWith('#') ? true : !href.includes('.'))) h = `${m[1]}.html${m[2] ?? ''}`;
        if (/^\s*(javascript|data|vbscript):/i.test(h)) h = '#';
        const ext = /^https?:/.test(h) ? ' rel="noopener"' : '';
        return `<a href="${esc(h)}"${title ? ` title="${esc(title)}"` : ''}${ext}>${text}</a>`;
      },
      table(token: Tokens.Table) {
        const cell = (c: Tokens.TableCell, tag: string) =>
          `<${tag}${c.align ? ` style="text-align:${c.align}"` : ''}>${this.parser.parseInline(c.tokens)}</${tag}>`;
        const head = `<tr>${token.header.map((c) => cell(c, 'th')).join('')}</tr>`;
        const rows = token.rows.map((r) => `<tr>${r.map((c) => cell(c, 'td')).join('')}</tr>`).join('');
        return `<div class="tablewrap"><table><thead>${head}</thead><tbody>${rows}</tbody></table></div>\n`;
      },
    },
  });
  const html = marked.parse(body) as string;
  return { html, headings };
}

function loadPages(): Page[] {
  const files = readdirSync(contentDir).filter((f) => f.endsWith('.md'));
  const slugs = new Set(files.map((f) => f.replace(/\.md$/, '')));
  return files
    .map((f) => {
      const slug = f.replace(/\.md$/, '');
      const { meta, body } = frontMatter(readFileSync(join(contentDir, f), 'utf8'));
      const expanded = expandMacros(body);
      const { html, headings } = renderPage(slug, expanded, slugs);
      const text = html
        .replace(/<[^>]+>/g, ' ')
        .replace(/&#39;|&quot;|&lt;|&gt;|&amp;/g, (e) => decode(e))
        .replace(/&[a-z0-9#]+;/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
      return {
        slug,
        title: meta.title ?? slug,
        group: meta.group ?? 'Reference',
        order: Number(meta.order ?? 100),
        eyebrow: meta.eyebrow,
        description: meta.description,
        body,
        headings,
        html,
        text,
      };
    })
    .filter((p) => p.group !== 'hidden')
    .sort((a, b) => GROUPS.indexOf(a.group) - GROUPS.indexOf(b.group) || a.order - b.order);
}

/* ---------- models table, generated from the registry ---------- */

const MB = 1024 * 1024;
function sizeOf(bytes?: number): string {
  if (!bytes) return 'n/a';
  return bytes >= 1024 * MB ? `${(bytes / 1024 / MB).toFixed(1)} GB` : `${Math.round(bytes / MB)} MB`;
}

function modelsData() {
  const aliases = registry.aliases();
  const byId: Record<string, string[]> = {};
  for (const [a, id] of Object.entries(aliases)) (byId[id] ??= []).push(a);
  return registry
    .list()
    .filter((m) => m.verb !== ('mock' as never))
    .map((m) => {
      const devices = [...new Set(m.variants.flatMap((v) => v.devices))];
      const bytes = Math.min(...m.variants.map((v) => v.bytes ?? Number.POSITIVE_INFINITY));
      return {
        id: m.id,
        verb: m.verb,
        accepts: m.accepts,
        params: m.params ?? '',
        size: Number.isFinite(bytes) ? sizeOf(bytes) : m.source && 'builtin' in m.source ? 'managed by Chrome' : 'n/a',
        mb: Number.isFinite(bytes) ? Math.round(bytes / MB) : 0,
        devices,
        status: m.status,
        license: m.license,
        aliases: byId[m.id] ?? [],
        features: m.features ?? [],
        notes: m.description ?? '',
        browserOnly: !!m.requires?.browser,
      };
    });
}

function modelsTable(): string {
  const rows = modelsData()
    .map((m) => {
      const dev = m.devices
        .map((d) => `<span class="chip ${d === 'webgpu' ? 'gpu' : 'wasm'}">${d === 'webgpu' ? 'WebGPU' : d === 'wasm' ? 'WASM' : 'CPU'}</span>`)
        .join(' ');
      const note = [m.aliases.map((a) => `alias <code>${esc(a)}</code>`).join(', '), esc(m.notes)].filter(Boolean).join(' · ');
      return `<tr data-verb="${esc(m.verb)}" data-accepts="${esc(m.accepts.join(' '))}" data-status="${esc(m.status)}"><td><div class="mid-id">${esc(m.id)}</div><div class="mnote">${esc(m.params)}${m.params ? ' params' : ''}</div>${note ? `<div class="mnote">${note}</div>` : ''}</td><td><code>${esc(m.verb)}</code></td><td>${m.accepts.map((a) => `<span class="chip in">${esc(a)}</span>`).join(' ')}</td><td><div class="sizebar"><span class="track"><span class="fill" style="width:${m.mb ? Math.max(4, Math.min(100, (Math.log10(m.mb) / Math.log10(3000)) * 100)).toFixed(0) : 0}%"></span></span><span class="v">${m.size}</span></div></td><td style="white-space:nowrap">${dev}</td><td><span class="chip ${esc(m.status)}">${esc(m.status)}</span></td><td><code>${esc(m.license)}</code></td></tr>`;
    })
    .join('\n');
  return `<div class="mfilters"><div class="seg" id="verbSeg" role="group" aria-label="Verb"></div><div class="seg" id="inSeg" role="group" aria-label="Input"></div><label class="chk"><input type="checkbox" id="stableOnly"> Stable only</label></div>
<p class="mcount" id="mcount"></p>
<div class="tablewrap"><table id="mtable" class="models"><thead><tr><th>Model</th><th>Verb</th><th>Accepts</th><th>Download</th><th>Runs on</th><th>Status</th><th>Licence</th></tr></thead><tbody>
${rows}
</tbody></table></div>`;
}

function verbTable(verb: string): string {
  const rows = modelsData()
    .filter((m) => m.verb === verb)
    .map(
      (m) =>
        `| [\`${m.id}\`](models) | ${m.accepts.join(', ')} | ${m.size} | ${m.status} | ${[m.aliases.map((a) => `\`${a}\``).join(', '), m.notes].filter(Boolean).join(' · ')} |`,
    )
    .join('\n');
  return `| Model | Accepts | Download | Status | Notes |\n| --- | --- | --- | --- | --- |\n${rows}`;
}

function aliasTable(): string {
  const rows = Object.entries(registry.aliases())
    .map(([a, id]) => `| \`${a}\` | \`${id}\` |`)
    .join('\n');
  return `| Alias | Resolves to |\n| --- | --- |\n${rows}`;
}

function expandMacros(body: string): string {
  return body
    .replace('{{models-table}}', () => modelsTable())
    .replace('{{alias-table}}', () => aliasTable())
    .replace(/\{\{models-([a-z]+)\}\}/g, (_, verb: string) => verbTable(verb))
    .replaceAll('{{version}}', pkg.version)
    .replaceAll('{{model-count}}', () => String(modelsData().length));
}

/* ---------- layout ---------- */

function nav(pages: Page[], current: string): string {
  return GROUPS.map((g) => {
    const items = pages.filter((p) => p.group === g);
    if (!items.length) return '';
    const li = items
      .map((p) => {
        const href = `${p.slug}.html`;
        const mod = p.eyebrow?.includes('→') && g === 'The six verbs' ? ` <span class="mod">${esc(p.eyebrow.split('·').pop()?.trim() ?? '')}</span>` : '';
        return `<li><a href="${href}"${p.slug === current ? ' aria-current="page"' : ''}>${esc(p.title)}${mod}</a></li>`;
      })
      .join('');
    return `<div class="navgroup"><h4>${g}</h4><ul>${li}</ul></div>`;
  }).join('\n');
}

const LOGO = `<svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true"><rect x="1" y="1" width="30" height="30" rx="8" fill="var(--accent)"/><path d="M9 22 L16 9 L23 22" fill="none" stroke="var(--accent-ink)" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/><circle cx="16" cy="19" r="2.4" fill="var(--accent-ink)"/></svg>`;

function layout(p: Page, pages: Page[]): string {
  const i = pages.indexOf(p);
  const link = (q: Page, cls: string, label: string) => `<a class="${cls}" href="${`${q.slug}.html`}"><small>${label}</small><b>${esc(q.title)}</b></a>`;
  const pager = `${i > 0 ? link(pages[i - 1], 'prev', '← Previous') : ''}${i < pages.length - 1 ? link(pages[i + 1], 'next', 'Next →') : ''}`;
  const toc = p.headings.map((h) => `<li><a href="#${h.id}">${esc(h.text)}</a></li>`).join('');
  const title = p.slug === 'intro' ? 'Edgewise · on-device AI for TypeScript' : `${p.title} · Edgewise`;
  const desc =
    p.description ?? 'Edgewise runs language, vision, speech, judging, embedding, image and forecasting models on the device, in browsers, Bun and Node.';
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>${esc(title)}</title>
<meta name="description" content="${esc(desc)}">
<link rel="icon" href="assets/icon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@500;600;700;800&family=Public+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap">
<link rel="stylesheet" href="assets/style.css">
${p.slug === 'demos' || p.slug === 'intro' ? `<meta name="ort-web" content="${ORT_WEB}">\n<script type="importmap">${IMPORT_MAP}</script>` : ''}
<script>try{const t=localStorage.getItem('ew-theme');if(t)document.documentElement.dataset.theme=t}catch(e){}</script>
</head>
<body data-page="${p.slug}">
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <button class="iconbtn" id="menuBtn" aria-label="Open navigation" aria-expanded="false"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
  <a class="brand" href="index.html" aria-label="Edgewise home">${LOGO}<b>edgewise</b></a>
  <span class="ver">v${pkg.version}</span>
  <nav class="toplinks" aria-label="Primary"><a href="index.html">Playground</a><a href="intro.html">Docs</a><a href="quickstart.html">Quickstart</a><a href="models.html">Models</a><a href="demos.html">Demos</a><a href="api/index.html">API</a><a href="https://github.com/kucukkanat/edgewise" rel="noopener">GitHub</a></nav>
  <div class="search"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg><input id="q" type="search" placeholder="Search docs" aria-label="Search docs" autocomplete="off"><div class="results" id="results" hidden></div></div>
  <button class="iconbtn" id="themeBtn" aria-label="Toggle dark mode"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/></svg></button>
</header>
<div class="shell">
<aside class="side" id="side" aria-label="Documentation">
${nav(pages, p.slug)}
</aside>
<main class="main" id="main">
<article class="page">
${p.eyebrow ? `<p class="eyebrow">${esc(p.eyebrow)}</p>\n` : ''}${p.html}
</article>
<nav class="pager" aria-label="Page navigation">${pager}</nav>
<p class="sitefoot">Edgewise ${pkg.version} · MIT licence · <a href="https://github.com/kucukkanat/edgewise/edit/main/docs/content/${p.slug}.md" rel="noopener">Edit this page</a></p>
</main>
<aside class="toc" aria-label="On this page">${toc ? `<h5>On this page</h5><ul id="tocList">${toc}</ul>` : ''}</aside>
</div>
<div class="scrim" id="scrim" hidden></div>
<script type="module" src="assets/app.js"></script>
${p.slug === 'demos' || p.slug === 'intro' ? '<script type="module" src="assets/demos.js"></script>' : ''}
</body>
</html>
`;
}

/* ---------- demo bundle: the real library, built for browsers ---------- */

// Bare imports the demo bundle leaves external, pinned to tested versions.
const IMPORT_MAP = JSON.stringify({
  imports: {
    phonemizer: 'https://cdn.jsdelivr.net/npm/phonemizer@1.2.1/+esm',
    zod: 'https://cdn.jsdelivr.net/npm/zod@4/+esm',
  },
});

const ORT_WEB = JSON.parse(readFileSync(join(root, '../node_modules/onnxruntime-web/package.json'), 'utf8')).version as string;
const TJS = JSON.parse(readFileSync(join(root, '../node_modules/@huggingface/transformers/package.json'), 'utf8')).version as string;

async function buildLib() {
  const r = await Bun.build({
    entrypoints: [join(root, 'lib-entry.ts')],
    outdir: join(out, 'lib'),
    target: 'browser',
    format: 'esm',
    // One file: code splitting with minified identifiers produced broken chunk exports in Bun 1.3.
    splitting: false,
    minify: true,
    sourcemap: 'linked',
    conditions: ['browser', 'source'],
    // Transformers.js and ONNX Runtime are bundled so both share one runtime instance.
    external: ['phonemizer', 'vgpu', 'vgpu/node', 'onnxruntime-node', 'sharp', 'zod'],
    naming: { entry: 'index.js' },
  });
  if (!r.success) throw new AggregateError(r.logs, 'demo bundle failed');
}

/* ---------- main ---------- */

rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, 'assets'), { recursive: true });
const pages = loadPages();
for (const p of pages) writeFileSync(join(out, `${p.slug}.html`), layout(p, pages));
writeFileSync(
  join(out, 'assets', 'search.json'),
  JSON.stringify(pages.map((p) => ({ slug: p.slug, title: p.title, group: p.group, headings: p.headings, text: p.text.slice(0, 6000) }))),
);
writeFileSync(join(out, 'assets', 'models.json'), JSON.stringify(modelsData()));
for (const f of readdirSync(join(root, 'theme'))) if (!f.startsWith('old-')) cpSync(join(root, 'theme', f), join(out, 'assets', f));
// The homepage: a live playground of every verb and model.
{
  const pg = join(root, 'playground');
  const html = readFileSync(join(pg, 'index.html'), 'utf8')
    .replaceAll('{{ort-web}}', ORT_WEB)
    .replaceAll('{{importmap}}', IMPORT_MAP)
    .replaceAll('{{version}}', pkg.version)
    .replaceAll('{{models-json}}', JSON.stringify(modelsData()).replace(/</g, '\\u003c'));
  writeFileSync(join(out, 'index.html'), html);
  mkdirSync(join(out, 'play'), { recursive: true });
  for (const f of readdirSync(pg)) if (f !== 'index.html') cpSync(join(pg, f), join(out, 'play', f), { recursive: true });
}
writeFileSync(join(out, '.nojekyll'), '');
writeFileSync(
  join(out, '404.html'),
  layout(
    {
      slug: '404',
      title: 'Not found',
      group: 'hidden',
      order: 0,
      body: '',
      headings: [],
      html: '<h1>Page not found</h1><p><a href="intro.html">Go to the introduction</a>.</p>',
      text: '',
    },
    pages,
  ).replace('<head>', `<head>\n<base href="${base}">`),
);
await buildLib();
if (withApi) {
  const proc = Bun.spawnSync(['bunx', 'typedoc', '--out', join(out, 'api')], { cwd: join(root, '..'), stdout: 'inherit', stderr: 'inherit' });
  if (proc.exitCode !== 0) throw new Error('typedoc failed');
} else if (!existsSync(join(out, 'api'))) {
  mkdirSync(join(out, 'api'));
  writeFileSync(join(out, 'api', 'index.html'), '<!doctype html><p>Run <code>bun docs/build.ts --api</code> to generate the API reference.</p>');
}
console.log(`built ${pages.length} pages into ${out}`);
