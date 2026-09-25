import json, numpy as np
def prefix(routes):
    body = "\n".join(f"- {r}" for r in routes) if routes else "- (none)"
    return f"Categories:\n{body}\n\nText:\n"
def cat_ranges(routes):
    ranges=[]; pos=len("Categories:\n")
    for r in routes:
        s=pos+2; e=s+len(r); ranges.append((s,e)); pos=e+1
    return ranges
def pools_offsets(tok, text, routes):
    """exact reference = model.route()"""
    p=prefix(routes); full=p+text
    enc=tok(full, return_offsets_mapping=True)
    offs=enc['offset_mapping']; n=len(offs)
    tp=np.zeros((1,n),np.float32); ts=len(p)
    idx=[i for i,(a,b) in enumerate(offs) if b>ts and a!=b]
    if idx: tp[0,idx]=1/len(idx)
    cp=np.zeros((len(routes),n),np.float32)
    for r,(s,e) in enumerate(cat_ranges(routes)):
        idx=[i for i,(a,b) in enumerate(offs) if a<e and b>s and a!=b]
        if idx: cp[r,idx]=1/len(idx)
    return np.array(enc['input_ids'],np.int64), tp, cp
def pools_bytes(tok_json_path, ids, text, routes, special_ids):
    """TS-friendly method: UTF-8 byte offsets from byte-level token string lengths"""
    vocab=json.load(open(tok_json_path))['model']['vocab']
    inv={v:k for k,v in vocab.items()}
    for t in json.load(open(tok_json_path))['added_tokens']: inv[t['id']]=t['content']
    p=prefix(routes)
    offs=[]; pos=0
    for i in ids:
        if i in special_ids: offs.append((pos,pos)); continue
        L=len(inv[i]); offs.append((pos,pos+L)); pos+=L
    blen=lambda s: len(s.encode('utf-8'))
    assert pos==blen(p+text), (pos, blen(p+text))
    n=len(ids); tp=np.zeros((1,n),np.float32); ts=blen(p)
    idx=[k for k,(a,b) in enumerate(offs) if b>ts and a!=b]
    if idx: tp[0,idx]=1/len(idx)
    cp=np.zeros((len(routes),n),np.float32)
    pos=blen("Categories:\n")
    for r,route in enumerate(routes):
        s=pos+2; e=s+blen(route); pos=e+1
        idx=[k for k,(a,b) in enumerate(offs) if a<e and b>s and a!=b]
        if idx: cp[r,idx]=1/len(idx)
    return tp, cp
