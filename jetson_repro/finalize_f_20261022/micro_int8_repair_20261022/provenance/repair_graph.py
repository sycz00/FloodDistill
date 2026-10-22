"""Zero-pad micro feature channels and densify groups; do not fit or recalibrate."""
import copy
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper


def transform(source, destination, width=16):
    model = onnx.load(source)
    graph = model.graph
    initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    producers = {o: n for n in graph.node for o in n.output}
    extra = []
    changes = []

    def constant(name):
        if name in initializers:
            return initializers[name]
        node = producers[name]
        if node.op_type == 'Constant':
            return numpy_helper.to_array(next(a.t for a in node.attribute if a.name == 'value'))
        raise ValueError('Expected constant: ' + name)

    def branch(name):
        chain = []
        while name not in initializers and producers[name].op_type in ('QuantizeLinear', 'DequantizeLinear'):
            node = producers[name]
            chain.insert(0, node)
            name = node.input[0]
        return constant(name), chain

    def add(value, name):
        graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    new_nodes = []
    for node in graph.node:
        node = copy.deepcopy(node)
        if node.op_type not in ('Conv', 'ConvTranspose'):
            new_nodes.append(node)
            continue
        tag = 'aligned_' + str(len(changes))
        weight, chain = branch(node.input[1])
        group = next((a.i for a in node.attribute if a.name == 'group'), 1)
        if node.op_type == 'Conv':
            cout, cin = weight.shape[0], weight.shape[1] * group
            dense = np.zeros((cout, cin) + weight.shape[2:], weight.dtype)
            for g in range(group):
                outs = slice(g * cout // group, (g + 1) * cout // group)
                ins = slice(g * cin // group, (g + 1) * cin // group)
                dense[outs, ins] = weight[outs]
            # The first convolution still consumes the exact six-band binding.
            new_cin = cin if node.name == '/stem/stem.0/Conv' else width
            shape = (width, new_cin) + weight.shape[2:]
        else:
            assert group == 1
            cin, cout = weight.shape[:2]
            dense = weight
            shape = (width, width) + weight.shape[2:]
        assert max(cin, cout) <= width
        padded = np.zeros(shape, weight.dtype)
        padded[tuple(slice(0, s) for s in dense.shape)] = dense
        feed = add(padded, tag + '_weight')
        for index, old in enumerate(chain):
            cloned = copy.deepcopy(old)
            cloned.name = tag + '_' + str(index) + '_' + old.op_type
            cloned.input[0] = feed
            axis = next((a.i for a in cloned.attribute if a.name == 'axis'), 1)
            for pos in (1, 2):
                value = constant(old.input[pos])
                if value.size > 1:
                    expanded = np.ones(padded.shape[axis], value.dtype) if pos == 1 else np.zeros(padded.shape[axis], value.dtype)
                    expanded[:value.size] = value
                    value = expanded
                cloned.input[pos] = add(value, cloned.name + '_parameter_' + str(pos))
            cloned.output[0] = cloned.name + '_output'
            feed = cloned.output[0]
            new_nodes.append(cloned)
        node.input[1] = feed
        if len(node.input) > 2:
            bias = constant(node.input[2])
            expanded = np.zeros(width, bias.dtype)
            expanded[:bias.size] = bias
            node.input[2] = add(expanded, tag + '_bias')
        for attr in node.attribute:
            if attr.name == 'group':
                attr.i = 1
        changes.append(dict(node=node.name, operator=node.op_type, old_channels=[cin, cout], old_groups=group,
                            new_channels=[shape[1], shape[0]] if node.op_type == 'Conv' else list(shape[:2]), new_groups=1))
        new_nodes.append(node)

    # Slice only the INT8 class dimension after the final shared-scale quantizer.
    original_output = graph.output[0].name
    final = next(n for n in new_nodes if original_output in n.output)
    assert final.op_type == 'QuantizeLinear'
    final.output[0] = 'aligned_logits_int8'
    new_nodes.append(helper.make_node('Slice', ['aligned_logits_int8',
        add(np.array([0], np.int64), 'aligned_slice_start'), add(np.array([2], np.int64), 'aligned_slice_end'),
        add(np.array([1], np.int64), 'aligned_slice_axis')], [original_output], name='aligned_two_class_slice'))
    # Dead weight branches must not leave unsupported grouped constants in the parser.
    needed = {o.name for o in graph.output}
    kept = []
    for node in reversed(new_nodes):
        if any(o in needed for o in node.output):
            kept.append(node)
            needed.update(node.input)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    kept_initializers = [i for i in graph.initializer if i.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists(), 'Preserve previous graph'
    onnx.save(model, destination)
    return changes
