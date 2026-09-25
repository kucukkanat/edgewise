import { builtinAliases, builtinManifests } from '../models/manifests.ts';
import { getConfig } from './config.ts';
import { ConfigError, ModelNotFoundError, UnsupportedInputError, WrongVerbError } from './errors.ts';
import type { Manifest, ManifestInput, ModelRef, PartType, Status, Verb } from './types.ts';

const models = new Map<string, Manifest>();
const aliases = new Map<string, string>();

for (const m of builtinManifests) models.set(m.id, Object.freeze(m));
for (const [alias, id] of Object.entries(builtinAliases)) aliases.set(alias, id);

export interface ListFilter {
  verb?: Verb;
  accepts?: PartType;
  status?: Status;
  /** Keep only models this device can run. Needs `capabilities()` first; see `registry.runnable`. */
  runnable?: boolean;
}

function resolveId(id: string): string {
  const seen = new Set<string>();
  let cur = id;
  while (aliases.has(cur)) {
    if (seen.has(cur)) throw new ConfigError(`Alias cycle at "${cur}".`);
    seen.add(cur);
    cur = aliases.get(cur) as string;
  }
  return cur;
}

function suggest(id: string): string {
  const all = [...models.keys(), ...aliases.keys()];
  const needle = id.toLowerCase().replace(/[^a-z0-9]/g, '');
  const close = all.filter((k) => {
    const hay = k.toLowerCase().replace(/[^a-z0-9]/g, '');
    return hay.includes(needle) || needle.includes(hay);
  });
  return close.length
    ? ` Did you mean ${close
        .slice(0, 3)
        .map((c) => `"${c}"`)
        .join(', ')}?`
    : '';
}

let runnableCheck: ((m: Manifest) => boolean) | null = null;

/** Set by the runtime once capabilities are known. */
export function setRunnableCheck(fn: (m: Manifest) => boolean): void {
  runnableCheck = fn;
}

export const registry = {
  /** Get a manifest by ID or alias. Throws `ModelNotFoundError` if unknown. */
  get(idOrAlias: string): Manifest {
    const id = resolveId(idOrAlias);
    const m = models.get(id);
    if (!m) {
      throw new ModelNotFoundError(`Unknown model "${idOrAlias}".${suggest(idOrAlias)}`, {
        hint: `"${idOrAlias}" is not in the registry. Check registry.list() for the models Edgewise supports.`,
      });
    }
    return m;
  },

  /** True when the ID or alias is known. */
  has(idOrAlias: string): boolean {
    return models.has(resolveId(idOrAlias));
  },

  /** List manifests, optionally filtered. */
  list(filter: ListFilter = {}): Manifest[] {
    return [...models.values()].filter((m) => {
      if (filter.verb && m.verb !== filter.verb) return false;
      if (filter.accepts && !m.accepts.includes(filter.accepts)) return false;
      if (filter.status && m.status !== filter.status) return false;
      if (filter.runnable && runnableCheck && !runnableCheck(m)) return false;
      return true;
    });
  },

  /** All aliases and the IDs they point to. */
  aliases(): Record<string, string> {
    return Object.fromEntries(aliases);
  },

  /** Point an alias at a model. */
  alias(alias: string, id: string): void {
    if (models.has(alias)) throw new ConfigError(`"${alias}" is a model ID and cannot be used as an alias.`);
    registry.get(id);
    aliases.set(alias, id);
  },
};

/** Register your own model. Returns the manifest, usable anywhere a model ID is. */
export function defineModel(input: ManifestInput): Manifest {
  if (!input.id || typeof input.id !== 'string') throw new ConfigError('defineModel needs an id.');
  if (aliases.has(input.id)) throw new ConfigError(`"${input.id}" is already an alias.`);
  if (!input.accepts?.length) throw new ConfigError(`Model "${input.id}" must list the inputs it accepts.`);
  const m: Manifest = Object.freeze({
    version: '1.0.0',
    status: 'stable' as const,
    license: 'unknown',
    variants: [{ dtype: 'fp32' as const, devices: ['webgpu', 'wasm', 'cpu'] as const }].map((v) => ({ ...v, devices: [...v.devices] })),
    ...input,
  });
  models.set(m.id, m);
  return m;
}

export interface ResolveOptions {
  verb: Verb;
  allowPreview?: boolean;
  /** Part types present in the input. */
  inputs?: PartType[];
}

/** Turn a model ref into a manifest and check that it fits this call. */
export function resolveManifest(ref: ModelRef, opts: ResolveOptions): Manifest {
  const m = typeof ref === 'string' ? registry.get(ref) : ref;
  if (m.verb !== opts.verb) {
    throw new WrongVerbError(`"${m.id}" is a ${m.verb} model and cannot be used with ${opts.verb}().`, {
      hint: `Use ${m.verb}() with "${m.id}", or pick a ${opts.verb} model from registry.list({ verb: '${opts.verb}' }).`,
    });
  }
  const cfg = getConfig();
  if (m.status !== 'stable' && !(opts.allowPreview ?? cfg.allowPreview)) {
    throw new ModelNotFoundError(`"${m.id}" is a ${m.status} model. Pass allowPreview: true to use it.`, {
      hint: `"${m.id}" is marked ${m.status}. Set allowPreview: true on the call, or configure({ allowPreview: true }).`,
    });
  }
  if (cfg.licenses.length && !cfg.licenses.includes(m.license)) {
    throw new ModelNotFoundError(`"${m.id}" has licence "${m.license}", which is not in configure({ licenses }).`);
  }
  for (const p of opts.inputs ?? []) {
    if (!m.accepts.includes(p)) {
      const alt = registry
        .list({ verb: opts.verb, accepts: p })
        .filter((x) => x.status === 'stable')
        .map((x) => x.id)[0];
      throw new UnsupportedInputError(`"${m.id}" does not accept ${p} input. It accepts ${m.accepts.join(', ')}.`, {
        hint: alt ? `"${m.id}" accepts ${m.accepts.join(', ')}. Try "${alt}" for ${p} input.` : `No ${opts.verb} model accepts ${p} input yet.`,
      });
    }
  }
  return m;
}
