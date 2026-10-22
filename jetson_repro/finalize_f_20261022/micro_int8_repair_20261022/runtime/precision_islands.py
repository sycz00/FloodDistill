"""Describe and verify B0's required floating attention/classifier islands."""
import hashlib,json,re
from pathlib import Path
from collections import defaultdict,deque

def contains_node(layer_name, node_name):
    # TensorRT may wrap fused names in PWN(...); reject prefix aliases MatMul/MatMul_1.
    return bool(re.search(r'(?<![A-Za-z0-9_./])'+re.escape(node_name)+r'(?![A-Za-z0-9_./])', layer_name))

def from_onnx(path):
    import onnx
    graph=onnx.load(str(path)).graph;byout={t:n for n in graph.node for t in n.output};weights={i.name:tuple(i.dims) for i in graph.initializer};consumers=defaultdict(list)
    for node in graph.node:
        for value in node.input:consumers[value].append(node)
    def shape(value):
        if value in weights:return weights[value]
        node=byout.get(value)
        if node is not None and node.op_type in ('QuantizeLinear','DequantizeLinear','Cast','Identity'):return shape(node.input[0])
        return ()
    matmuls=[n for n in graph.node if n.op_type=='MatMul'];heads=[n for n in graph.node if n.op_type=='Conv' and shape(n.input[1]) and shape(n.input[1])[0]==2]
    if len(matmuls)!=8 or len(heads)!=1:raise ValueError('Expected8 B0 attention MatMuls and one two-class Conv')
    selected={n.name:n for n in matmuls};pending=deque(v for n in matmuls for v in n.output);seen=set();divs=set()
    while pending:
        value=pending.popleft()
        if value in seen:continue
        seen.add(value)
        for node in consumers[value]:
            if node.op_type=='Conv':raise ValueError('Attention reaches Conv before its normalization Div')
            selected[node.name]=node
            if node.op_type=='Div':divs.add(node.name)
            else:pending.extend(node.output)
    if len(divs)!=4:raise ValueError('Expected four complete attention normalization islands')
    selected[heads[0].name]=heads[0]
    return dict(schema_version=1,policy='FP32 attention MatMuls through normalization Div and floating two-class classifier; TF32 disabled; remaining FP16/INT8 placement audited separately',node_names=sorted(selected),tensor_names=sorted({v for n in selected.values() for v in n.output}),required_compute_nodes=sorted([n.name for n in matmuls]+list(divs)+[heads[0].name]),attention_matmuls=8,normalizers=4,classifier=heads[0].name)

def constrain(network,config,trt,spec):
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS);wanted=set(spec['node_names']);tensors=set(spec['tensor_names']);matched=[];names=[]
    for i in range(network.num_layers):
        layer=network.get_layer(i);outputs=[layer.get_output(j) for j in range(layer.num_outputs)]
        if layer.name not in wanted and not any(t.name in tensors for t in outputs):continue
        if str(layer.type).split('.')[-1] in ('CONSTANT','SHAPE','QUANTIZE','DEQUANTIZE'):continue
        layer.precision=trt.float32
        for j,tensor in enumerate(outputs):
            if not tensor.is_shape_tensor and tensor.dtype not in (trt.int32,trt.bool):layer.set_output_type(j,trt.float32)
        matched.append(dict(name=layer.name,outputs=[t.name for t in outputs]));names.append(layer.name)
    missing=[name for name in spec['required_compute_nodes'] if not any(contains_node(seen,name) for seen in names)]
    if missing:raise ValueError('Cannot constrain required FP32 layers: '+repr(missing))
    return dict(required=spec['required_compute_nodes'],constrained=matched)

def inspect(layers,spec):
    rows=layers.get('Layers',[]) if isinstance(layers,dict) else layers;missing=[];bad=[];verified=[]
    for name in spec['required_compute_nodes']:
        matching=[r for r in rows if contains_node(r.get('Name',''),name)]
        if not matching:missing.append(name);continue
        okay=True
        for row in matching:
            formats=[str(t.get('Format/Datatype','')) for t in row.get('Outputs',[])]
            if not formats or any(re.search(r'FP16|Half|BF16|INT8',v,re.I) for v in formats) or not any(re.search(r'FP32|Float',v,re.I) for v in formats):okay=False;bad.append(dict(node=name,layer=row.get('Name'),formats=formats))
        if okay:verified.append(name)
    return dict(passed=not missing and not bad,verified=verified,missing=missing,wrong_or_unknown=bad,scope='Requested FP32 compute and inspected output formats; does not certify every internal kernel arithmetic operation')


def validate_certificate(record, model, root, layers=None):
    """Reject missing, stale or failed B0 precision certificates before observation."""
    if record.get('status') != 'built':
        raise ValueError('Build is not certified as built')
    if record.get('model_sha256') != model['onnx_sha256']:
        raise ValueError('Build/model identity mismatch')
    path = Path(root) / model['fp32_islands']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if model.get('fp32_islands_sha256') != digest or record.get('fp32_islands_sha256') != digest:
        raise ValueError('Missing or stale FP32 island policy identity')
    spec = json.loads(path.read_text())
    audit = record.get('fp32_island_audit', {})
    if not audit.get('passed') or set(audit.get('verified', [])) != set(spec['required_compute_nodes']):
        raise ValueError('Missing or failed FP32 island certificate')
    if record.get('fp32_constraints', {}).get('required') != spec['required_compute_nodes']:
        raise ValueError('Required FP32 compute constraints were not recorded')
    if layers is not None and not inspect(layers, spec)['passed']:
        raise ValueError('Saved engine inspector contradicts FP32 certificate')
    return True
