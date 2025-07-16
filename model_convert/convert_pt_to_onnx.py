import torch
import os

def convert_pt_to_onnx(pt_path: str, onnx_path: str = None, input_size=(640, 640)):
    """
    将 yolov8n/s/m/x.pt 模型转换为 onnx 格式。
    :param pt_path: 输入的 .pt 文件路径
    :param onnx_path: 输出的 .onnx 文件路径（可选，默认为同名 .onnx）
    :param input_size: 输入尺寸，默认为 (640, 640)
    """
    # 尝试导入 ultralytics（yolov8 官方库）
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请先安装 ultralytics 库: pip install ultralytics")

    if onnx_path is None:
        onnx_path = os.path.splitext(pt_path)[0] + ".onnx"

    # ultralytics 的 name 参数只支持文件名，不支持路径
    export_name = os.path.splitext(os.path.basename(onnx_path))[0]
    export_dir = os.path.dirname(onnx_path) or os.getcwd()

    # 加载模型
    model = YOLO(pt_path)

    # 导出为 onnx
    model.export(format="onnx", imgsz=input_size, dynamic=False, simplify=True, optimize=True, half=False, device="cpu", 
                 opset=12, 
                 name=export_name)
    # 移动导出文件到目标路径（ultralytics 默认导出到当前目录）
    src_onnx = os.path.join(os.getcwd(), export_name + ".onnx")
    dst_onnx = os.path.join(export_dir, export_name + ".onnx")
    if os.path.abspath(src_onnx) != os.path.abspath(dst_onnx):
        import shutil
        shutil.move(src_onnx, dst_onnx)
    print(f"模型已导出为: {dst_onnx}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="将 yolov8n/s/m/x.pt 转换为 onnx 格式")
    parser.add_argument("pt_path", type=str, help="输入的 .pt 文件路径")
    parser.add_argument("--onnx_path", type=str, default=None, help="输出的 .onnx 文件路径")
    parser.add_argument("--input_size", type=int, nargs=2, default=[640, 640], help="输入尺寸，默认 640 640")
    args = parser.parse_args()
    convert_pt_to_onnx(args.pt_path, args.onnx_path, tuple(args.input_size)) 