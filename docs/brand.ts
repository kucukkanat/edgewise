/**
 * Builds the site's brand images from the source art in docs/brand/ into docs/theme/ (served as assets/).
 *
 *   bun docs/brand.ts
 *
 * raven.webp       the full raven, trimmed, for the homepage and docs heroes
 * raven-head.webp  the head, trimmed, for the README, the 404 page and empty states
 * og.png           the 1200×630 social card
 * icon-180.png     the touch icon, from icon.svg
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import sharp from 'sharp';

const root = import.meta.dir;
const src = (f: string) => join(root, 'brand', f);
const out = (f: string) => join(root, 'theme', f);

// Twice the largest size each image is shown at, so it stays sharp on high-density screens.
await sharp(src('raven.webp')).trim().resize({ height: 640 }).webp({ quality: 82, alphaQuality: 90 }).toFile(out('raven.webp'));
await sharp(src('raven-head.webp')).trim().resize({ width: 560 }).webp({ quality: 82, alphaQuality: 90 }).toFile(out('raven-head.webp'));

const W = 1200;
const H = 630;
const card = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0B1018"/><stop offset="1" stop-color="#18233A"/></linearGradient>
    <radialGradient id="glow" cx="0.8" cy="0.3" r="0.42"><stop offset="0" stop-color="#FFB13B" stop-opacity=".22"/><stop offset="1" stop-color="#FFB13B" stop-opacity="0"/></radialGradient>
  </defs>
  <rect width="${W}" height="${H}" fill="url(#bg)"/>
  <rect width="${W}" height="${H}" fill="url(#glow)"/>
  <g font-family="Helvetica Neue, Helvetica, Arial, sans-serif">
    <text x="72" y="150" fill="#FFB13B" font-size="30" font-weight="700" letter-spacing="1">edgewise</text>
    <text x="72" y="262" fill="#EEF3F8" font-size="68" font-weight="800">Six verbs for</text>
    <text x="72" y="342" fill="#EEF3F8" font-size="68" font-weight="800">on-device AI.</text>
    <text x="72" y="416" fill="#AFBDD0" font-size="28">Language, vision, speech, judging,</text>
    <text x="72" y="454" fill="#AFBDD0" font-size="28">embeddings, images and forecasts</text>
    <text x="72" y="492" fill="#AFBDD0" font-size="28">in the browser, Bun and Node.</text>
    <text x="72" y="566" fill="#7F8DA3" font-size="24" font-family="Menlo, monospace">bun add edgewise</text>
  </g>
</svg>`);
// The whole head fits, beak included: the beak is what makes it a raven.
const head = await sharp(src('raven-head.webp')).trim().resize({ height: 540 }).toBuffer();
const { width = 0 } = await sharp(head).metadata();
await sharp(card)
  .composite([{ input: head, left: W - width - 48, top: 52 }])
  .png({ compressionLevel: 9 })
  .toFile(out('og.png'));

await sharp(readFileSync(out('icon.svg')), { density: 600 })
  .resize(180, 180, { fit: 'contain', background: { r: 244, g: 246, b: 249, alpha: 1 } })
  .png()
  .toFile(out('icon-180.png'));

console.log('brand images written to docs/theme/');
