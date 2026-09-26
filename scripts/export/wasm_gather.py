"""Rewrite GatherBlockQuantized (a contrib op missing from ONNX Runtime Web's WebAssembly build) into
standard ONNX ops that gather the packed 4-bit rows and unpack only those rows. Weights stay 4-bit."""
import sys, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto as T

def rewrite(m):
    g = m.graph
    init = {i.name: i for i in g.initializer}
    out, k = [], 0
    def c(name, arr):
        g.initializer.append(numpy_helper.from_array(np.asarray(arr), name)); return name
    for n in g.node:
        if n.op_type != 'GatherBlockQuantized':
            out.append(n); continue
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        assert a.get('bits', 4) == 4 and a.get('gather_axis', 0) == 0, a
        q_name, ids, s_name = n.input[0], n.input[1], n.input[2]
        zp_name = n.input[3] if len(n.input) > 3 and n.input[3] else None
        q = init[q_name]; s = init[s_name]
        V, Dh = q.dims; D = Dh * 2; B = a['block_size']; NB = D // B
        assert a['quantize_axis'] in (1, -1) and len(q.dims) == 2, (a, q.dims)
        p = f'/gbq{k}/'; k += 1
        stype = s.data_type
        node = lambda op, ins, outs, **kw: out.append(helper.make_node(op, ins, outs, name=p + outs[0].split('/')[-1], **kw))
        four = c(p + 'four', np.array(4, dtype=np.uint8))
        neg1 = c(p + 'neg1', np.array([-1], dtype=np.int64))
        def unpack(src, tag):
            # low nibble: shift left then right, so only BitShift (opset 11) is needed
            node('BitShift', [src, four], [p + tag + 'sl'], direction='LEFT')
            node('BitShift', [p + tag + 'sl', four], [p + tag + 'lo'], direction='RIGHT')
            node('BitShift', [src, four], [p + tag + 'hi'], direction='RIGHT')
            node('Unsqueeze', [p + tag + 'lo', neg1], [p + tag + 'lo1'])
            node('Unsqueeze', [p + tag + 'hi', neg1], [p + tag + 'hi1'])
            node('Concat', [p + tag + 'lo1', p + tag + 'hi1'], [p + tag + 'pair'], axis=-1)
            return p + tag + 'pair'
        # shape of the ids, to rebuild [..., D]
        node('Shape', [ids], [p + 'idshape'])
        node('Gather', [q_name, ids], [p + 'qrows'], axis=0)                 # [..., D/2]
        pair = unpack(p + 'qrows', 'q')                                          # [..., D/2, 2]
        blocks = c(p + 'blocks', np.array([NB, B], dtype=np.int64))
        node('Concat', [p + 'idshape', blocks], [p + 'bshape'], axis=0)
        node('Reshape', [pair, p + 'bshape'], [p + 'qb'])                      # [..., NB, B]
        node('Cast', [p + 'qb'], [p + 'qf'], to=stype)
        node('Gather', [s_name, ids], [p + 'srows'], axis=0)                 # [..., NB]
        axes_last = c(p + 'axlast', np.array([-1], dtype=np.int64))
        node('Unsqueeze', [p + 'srows', axes_last], [p + 's1'])
        if zp_name:
            node('Gather', [zp_name, ids], [p + 'zrows'], axis=0)             # [..., ceil(NB/2)]
            zpair = unpack(p + 'zrows', 'z')
            zlen = c(p + 'zlen', np.array([-1], dtype=np.int64))
            node('Concat', [p + 'idshape', zlen], [p + 'zshape'], axis=0)
            node('Reshape', [zpair, p + 'zshape'], [p + 'zflat'])
            st = c(p + 'z0', np.array([0], dtype=np.int64)); en = c(p + 'znb', np.array([NB], dtype=np.int64))
            node('Slice', [p + 'zflat', st, en, axes_last], [p + 'zs'])
            node('Cast', [p + 'zs'], [p + 'zf'], to=stype)
        else:
            node('Cast', [c(p + 'z8', np.array(8, dtype=np.uint8))], [p + 'zf'], to=stype)
        node('Unsqueeze', [p + 'zf', axes_last], [p + 'z1']) if zp_name else None
        node('Sub', [p + 'qf', p + 'z1' if zp_name else p + 'zf'], [p + 'centered'])
        node('Mul', [p + 'centered', p + 's1'], [p + 'deq'])
        dD = c(p + 'dD', np.array([D], dtype=np.int64))
        node('Concat', [p + 'idshape', dD], [p + 'oshape'], axis=0)
        out.append(helper.make_node('Reshape', [p + 'deq', p + 'oshape'], [n.output[0]], name=p + 'out'))
        print('rewrote', n.name, f'V={V} D={D} block={B} zp={bool(zp_name)}')
    del g.node[:]; g.node.extend(out)
    return m

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    m = rewrite(onnx.load(src))
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=dst.split('/')[-1] + '_data', size_threshold=1024)
    print('saved', dst)
