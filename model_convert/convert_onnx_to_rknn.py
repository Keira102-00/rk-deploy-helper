def convert_onnx_to_rknn(onnx_path: str, rknn_path: str = None, input_size=(640, 640)):
    """
    使用 RKNNToolkitLite 将 onnx 文件转换为 rknn 文件。
    :param onnx_path: 输入的 onnx 文件路径
    :param rknn_path: 输出的 rknn 文件路径（可选，默认为同名 .rknn）
    :param input_size: 输入尺寸，默认为 (640, 640)
    """
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        raise ImportError("请先安装 rknn-toolkit2: pip install rknn-toolkit2")

    import os
    if rknn_path is None:
        rknn_path = os.path.splitext(onnx_path)[0] + ".rknn"

    rknn = RKNNLite()
    print(f"开始加载 ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path, inputs=["input"], input_size_list=[list(input_size) + [3]])
    if ret != 0:
        print("ONNX 加载失败！")
        return
    print("ONNX 加载成功，开始构建...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("RKNN 构建失败！")
        return
    print("构建成功，导出 rknn 文件...")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print("RKNN 导出失败！")
        return
    print(f"RKNN 文件已导出: {rknn_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="将 onnx 转换为 rknn 格式")
    parser.add_argument("onnx_path", type=str, help="输入的 onnx 文件路径")
    parser.add_argument("--rknn_path", type=str, default=None, help="输出的 rknn 文件路径")
    parser.add_argument("--input_size", type=int, nargs=2, default=[640, 640], help="输入尺寸，默认 640 640")
    args = parser.parse_args()
    convert_onnx_to_rknn(args.onnx_path, args.rknn_path, tuple(args.input_size)) 