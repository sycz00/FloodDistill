import onnxruntime as ort

def session(path,semantic=False):
 opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
 if semantic:opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
 return ort.InferenceSession(str(path),opt,providers=['CPUExecutionProvider'])
