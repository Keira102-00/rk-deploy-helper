import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="RKNN 推理命令行工具")
    parser.add_argument("--rknn_path", type=str, required=True, help="RKNN 模型路径")
    parser.add_argument("--mode", type=str, choices=["video", "camera"], required=True, help="推理方式：video 或 camera")
    parser.add_argument("--input", type=str, default=None, help="输入源：视频文件路径或摄像头ID（camera 模式下可选，默认为0）")
    parser.add_argument("--input_size", type=int, nargs=2, default=[640, 640], help="输入尺寸，默认 640 640")
    args = parser.parse_args()

    if args.mode == "video":
        if args.input is None:
            print("video 模式下必须指定 --input 视频文件路径")
            sys.exit(1)
        from inference.video_inference import RKNNVideoInfer
        infer = RKNNVideoInfer(args.rknn_path, tuple(args.input_size))
        infer.run_on_video(args.input)
    elif args.mode == "camera":
        camera_id = 0
        if args.input is not None:
            try:
                camera_id = int(args.input)
            except ValueError:
                print("camera 模式下 --input 应为摄像头ID（整数）")
                sys.exit(1)
        from inference.camera_inference import RKNNCameraInfer
        infer = RKNNCameraInfer(args.rknn_path, tuple(args.input_size))
        infer.run_on_camera(camera_id)
    else:
        print("未知推理方式")
        sys.exit(1)

if __name__ == "__main__":
    main() 