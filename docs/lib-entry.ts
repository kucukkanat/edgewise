// The browser bundle behind the live demos. A default export of the namespaces keeps Bun's bundler
// from dropping re-exports of a package marked "sideEffects": false.
import * as helpers from '../src/helpers/index.ts';
import * as edgewise from '../src/index.ts';
import * as worker from '../src/worker/index.ts';

export default { ...edgewise, helpers, worker };
