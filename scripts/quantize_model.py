from onnxruntime.quantization import quantize_dynamic, QuantType
import os

src = '/build/model/model.onnx'
dst = '/build/model/model_q.onnx'

if os.path.exists(src):
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    os.replace(dst, src)
    print('Quantization complete')
else:
    print(f'WARNING: {src} not found, skipping quantization')