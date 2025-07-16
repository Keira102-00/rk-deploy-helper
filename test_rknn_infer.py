from rknn.api import RKNN
import numpy as np
import sys

if len(sys.argv) < 2:
    print("用法: python test_rknn_infer.py <rknn_path>")
    sys.exit(1)

rknn_path = sys.argv[1]
rknn = RKNN()
ret = rknn.load_rknn(rknn_path)
if ret != 0:
    print("RKNN 加载失败！")
    exit(1)
ret = rknn.init_runtime()
if ret != 0:
    print("RKNN 运行环境初始化失败！")
    exit(1)
# 假设输入为(1, 3, 640, 640)
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = rknn.inference(inputs=[dummy_input])
print("RKNN输出:", [o.shape if hasattr(o, 'shape') else type(o) for o in outputs]) 