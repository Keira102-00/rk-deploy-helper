import onnxruntime as ort
import numpy as np
import sys

if len(sys.argv) < 2:
    print("用法: python test_onnx_infer.py <onnx_path>")
    sys.exit(1)

onnx_path = sys.argv[1]
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"ONNX输入名: {input_name}, 输入shape: {input_shape}")

# 生成随机输入
input_data = np.random.randn(*[d if d else 1 for d in input_shape]).astype(np.float32)
outputs = session.run(None, {input_name: input_data})
print("ONNX输出shape:", [o.shape for o in outputs]) 