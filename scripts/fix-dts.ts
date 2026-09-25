// Declaration files keep the `.ts` specifiers used in the sources; published types point at `.js`.
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : p.endsWith('.d.ts') ? [p] : [];
  });
}
let n = 0;
for (const f of walk('dist')) {
  const src = readFileSync(f, 'utf8');
  const out = src.replace(/(from\s+|import\()(['"])(\.{1,2}\/[^'"]+?)\.ts\2/g, '$1$2$3.js$2');
  if (out !== src) {
    writeFileSync(f, out);
    n++;
  }
}
console.log(`fixed ${n} declaration files`);
