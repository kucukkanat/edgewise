"""Make WebAssembly-friendly copies of 4-bit ONNX models.

ONNX Runtime Web's WebAssembly build has no GatherBlockQuantized kernel, so 4-bit exports with quantized
embeddings fail there and Edgewise had to fall back to fp16, which ONNX Runtime upcasts to fp32 in memory
(about 3 GB for a 350M model). wasm_gather.rewrite() replaces that op with standard ops that gather the
packed rows and unpack only those, so weights stay 4-bit and outputs are bit-identical.

  python wasm_q4.py <repo> <revision> <out-dir> <file> [<file> …]   # e.g. onnx/model_q4.onnx

Each file is downloaded from $HUB (default https://huggingface.co), rewritten, checked against the original
on native ONNX Runtime, and written to <out-dir> with its external data. SHA-256 sums are printed for the
manifest.
"""
import hashlib, os, sys, urllib.request
import numpy as np, onnx, onnxruntime as ort
from wasm_gather import rewrite

HUB = os.environ.get('HUB', 'https://huggingface.co').rstrip('/')

def fetch(url, dst):
    if os.path.exists(dst): return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst + '.part', 'wb') as f:
        while chunk := r.read(1 << 22): f.write(chunk)
    os.rename(dst + '.part', dst)

def feeds_for(s):
    rng = np.random.default_rng(0)
    seq, out = 8, {}
    for i in s.get_inputs():
        def dim(d, k):
            if isinstance(d, int): return d
            d = str(d)
            if 'batch' in d: return 1
            if 'past' in d: return 0
            if 'total' in d or 'sequence' in d or d == 'seq': return seq
            if 'num_image_tokens' in d or 'image' in d: return 1
            return 1
        shape = [dim(d, k) for k, d in enumerate(i.shape)]
        if 'int64' in i.type:
            if i.name == 'input_ids': out[i.name] = rng.integers(10, 5000, size=shape, dtype=np.int64)
            elif 'position' in i.name: out[i.name] = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
            elif 'num_logits' in i.name: out[i.name] = np.array(1, dtype=np.int64)
            else: out[i.name] = np.ones(shape, dtype=np.int64)
        else:
            dt = np.float16 if 'float16' in i.type else np.float32
            out[i.name] = (rng.standard_normal(shape) * 0.1).astype(dt) if 'past' not in i.name else np.zeros(shape, dt)
    return out

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(1 << 22): h.update(chunk)
    return h.hexdigest()

repo, rev, out_dir, *files = sys.argv[1:]
work = os.path.join(out_dir, '.orig')
for rel in files:
    name = os.path.basename(rel)
    src = os.path.join(work, name)
    fetch(f'{HUB}/{repo}/resolve/{rev}/{rel}', src)
    m = onnx.load(src, load_external_data=False)
    locs = {e.value for t in m.graph.initializer for e in t.external_data if e.key == 'location'}
    for loc in locs: fetch(f'{HUB}/{repo}/resolve/{rev}/{os.path.dirname(rel)}/{loc}', os.path.join(work, loc))
    m = onnx.load(src)
    if not any(n.op_type == 'GatherBlockQuantized' for n in m.graph.node):
        print(f'{name}: no GatherBlockQuantized, nothing to do'); continue
    rewrite(m)
    dst = os.path.join(out_dir, name)
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=name + '_data', size_threshold=1024)
    a = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    b = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
    f = feeds_for(a)
    ra, rb = a.run(None, f), b.run(None, f)
    worst = max(float(np.abs(x.astype(np.float32) - y.astype(np.float32)).max()) if x.size else 0.0 for x, y in zip(ra, rb))
    print(f'{name}: parity max abs diff {worst}')
    assert worst < 1e-3, 'outputs differ'
    for p in (dst, dst + '_data'): print(f'  {os.path.basename(p)}  {os.path.getsize(p)}  sha256 {sha(p)}')
