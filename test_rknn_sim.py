from rknn.api import RKNN
import numpy as np
import sys

if len(sys.argv) < 2:
    print("用法: python test_rknn_sim.py /path/to/model.onnx")
    sys.exit(1)

onnx_path = sys.argv[1]
rknn = RKNN()
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588'
)
ret = rknn.load_onnx(model=onnx_path)
if ret != 0:
    print("ONNX 加载失败！")
    exit(1)
ret = rknn.build(do_quantization=False)
if ret != 0:
    print("RKNN 构建失败！")
    exit(1)
ret = rknn.init_runtime(target=None)  # PC仿真
if ret != 0:
    print("RKNN 运行环境初始化失败！")
    exit(1)
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = rknn.inference(inputs=[dummy_input])
print("RKNN仿真输出:", [o.shape if hasattr(o, 'shape') else type(o) for o in outputs]) 