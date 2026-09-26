// Runs the voice assistant's language model off the main thread, so voice activity detection and the UI
// stay responsive while it thinks. Uses Edgewise's own worker mode.
import lib from '../lib/index.js';

lib.worker.serveWorker();
