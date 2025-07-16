# 测试脚本说明

## 1. ONNX 模型推理仿真

```bash
python test_onnx_infer.py /path/to/model.onnx
```
- 验证 onnx 文件能否正常推理。

## 2. RKNN 模型仿真推理

```bash
python test_rknn_infer.py /path/to/model.rknn
```
- 验证 rknn 文件能否被正确加载和推理。

## 3. 前后处理与可视化测试

```bash
python test_utils.py
```
- 会生成一张带有检测框的测试图片 `test_utils_result.jpg`。

## 4. 代码结构和自动化脚本完善
- 可根据实际需求补充更多测试脚本。
- 建议所有测试脚本均可通过命令行参数指定模型路径。 