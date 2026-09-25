---
title: Live demos
group: Reference
order: 1
description: Run Edgewise models in your browser. Nothing leaves the page.
---

# Live demos

These demos run the real library in your browser. Models download once from Hugging Face and are cached; nothing you type or say leaves the page. Each button says how much it downloads.

<div class="demo" id="demo-generate">
<h2 id="generate-text">generate: text</h2>
<p>LFM2.5 350M writes a reply, streamed token by token.</p>
<textarea id="gen-in" rows="2">Give me three names for a bakery that sells sourdough.</textarea>
<div class="demo-row"><button class="btn primary" data-run="generate">Run · 300–725 MB</button><span class="demo-status"></span></div>
<pre class="demo-out" id="gen-out"></pre>
</div>

<div class="demo" id="demo-evaluate">
<h2 id="evaluate-route">evaluate: route a prompt</h2>
<p>A zero-shot NLI judge scores the text against each lane.</p>
<textarea id="ev-in" rows="2">My invoice from March charged me twice.</textarea>
<div class="demo-row"><button class="btn primary" data-run="evaluate">Run · ~90 MB</button><span class="demo-status"></span></div>
<div class="bars" id="ev-out"></div>
</div>

<div class="demo" id="demo-embed">
<h2 id="embed-similarity">embed: similarity</h2>
<p>MiniLM embeds each line; the bars show how close each is to the first.</p>
<textarea id="em-in" rows="4">How do I reset my password?
I forgot my login details
Where is your office?
Best pizza in Naples</textarea>
<div class="demo-row"><button class="btn primary" data-run="embed">Run · ~23 MB</button><span class="demo-status"></span></div>
<div class="bars" id="em-out"></div>
</div>

<div class="demo" id="demo-speak">
<h2 id="speak-tts">speak: text to speech</h2>
<p>Kokoro 82M speaks, starting with the first sentence.</p>
<textarea id="sp-in" rows="2">Hello from Edgewise. This voice was made in your browser.</textarea>
<div class="demo-row"><select id="sp-voice"><option>af_heart</option><option>af_bella</option><option>am_michael</option><option>bf_emma</option><option>bm_george</option></select><button class="btn primary" data-run="speak">Speak · ~90–330 MB</button><span class="demo-status"></span></div>
</div>

<div class="demo" id="demo-transcribe">
<h2 id="generate-speech">generate: speech to text</h2>
<p>Record a few seconds; Moonshine Tiny transcribes it.</p>
<div class="demo-row"><button class="btn primary" data-run="transcribe">Hold to talk · ~30 MB</button><span class="demo-status"></span></div>
<pre class="demo-out" id="tr-out"></pre>
</div>

<div class="demo" id="demo-forecast">
<h2 id="forecast-series">forecast: a time series</h2>
<p>Chronos-Bolt Tiny continues a noisy daily pattern, with an 80% range.</p>
<div class="demo-row"><button class="btn primary" data-run="forecast">Run · ~35 MB</button><span class="demo-status"></span></div>
<svg id="fc-out" viewBox="0 0 640 230" role="img" aria-label="Forecast chart"></svg>
</div>

<div class="demo" id="demo-paint">
<h2 id="paint-image">paint: text to image</h2>
<p>SD-Turbo paints a 512×512 image in one step. It needs WebGPU and downloads about 2.5 GB.</p>
<textarea id="pt-in" rows="2">a lighthouse on a cliff at sunset, watercolor</textarea>
<div class="demo-row"><button class="btn primary" data-run="paint">Paint · ~2.5 GB</button><span class="demo-status"></span></div>
<canvas id="pt-out" width="512" height="512"></canvas>
</div>
