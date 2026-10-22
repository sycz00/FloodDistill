import copy,json,sys
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper
from check_repair_support import session
P=Path(__file__).resolve().parent
D=Path('/home/schmalstieg/repos/FloodDistill/jetson_repro/finalize_f_20261022/device_handoff')
sys.path.insert(0,str(D/'runtime'))
import dataset_io
from device_runtime import encode
dataset_io.ROOT=Path('/home/schmalstieg/repos/MajorRev/revision_package_20261021/8_jetson_export_20261022')
def unrounded(path,out):
 m=onnx.load(path);g=m.graph;aliases={};nodes=[];inits={x.name:numpy_helper.to_array(x) for x in g.initializer}
 lead=next(n for n in g.node if n.op_type=='DequantizeLinear' and n.input[0]=='image')
 scale=inits[lead.input[1]];assert scale.size==1
 g.initializer.append(numpy_helper.from_array(scale,'structural_input_scale'))
 nodes=[helper.make_node('Cast',['image'],['structural_input_float'],to=onnx.TensorProto.FLOAT),helper.make_node('Mul',['structural_input_float','structural_input_scale'],['structural_input_dequantized'])]
 for n in g.node:
  if n is lead or (n.name==lead.name):aliases[n.output[0]]='structural_input_dequantized';continue
  n=copy.deepcopy(n);n.input[:]=[aliases.get(i,i) for i in n.input]
  if n.op_type in ('QuantizeLinear','DequantizeLinear'):aliases[n.output[0]]=n.input[0];continue
  nodes.append(n)
 for o in g.output:o.name=aliases.get(o.name,o.name);o.type.tensor_type.elem_type=onnx.TensorProto.FLOAT
 del g.node[:];g.node.extend(nodes);needed={x for n in nodes for x in n.input};kept=[i for i in g.initializer if i.name in needed];del g.initializer[:];g.initializer.extend(kept);del g.value_info[:]
 onnx.checker.check_model(m);onnx.save(m,out)
for d in sorted((P/'models').iterdir()):
 report=json.loads((d/'REPAIR_VALIDATION.json').read_text());e=report['source_model'];io=json.loads((D/e['io']).read_text())
 unrounded(D/e['onnx'],d/'old_unrounded.onnx');unrounded(d/'model.onnx',d/'new_unrounded.onnx')
 a=session(d/'old_unrounded.onnx',True);b=session(d/'new_unrounded.onnx',True);rows=[]
 for image,mask,rec in dataset_io.source('s11val'):
  x=encode(image[None],io['host_input_scale'],0);left=a.run(None,{'image':x})[0];right=b.run(None,{'image':x})[0]
  rows.append(dict(name=rec['name'],passed=bool(np.allclose(left,right,atol=.001,rtol=.001)),max_abs=float(np.max(np.abs(left-right)))))
 report['unrounded_all89']=dict(atol=.001,rtol=.001,passed=all(r['passed'] for r in rows),rows=rows)
 # Integer exactness remains separate: a structural pass does not override QDQ differences.
 (d/'REPAIR_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
 print(e['seed'],report['unrounded_all89']['passed'],max(r['max_abs'] for r in rows),flush=True)
 assert report['unrounded_all89']['passed']
