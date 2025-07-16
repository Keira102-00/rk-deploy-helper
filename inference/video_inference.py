import cv2
import numpy as np
from utils import preprocess, postprocess, draw_boxes

class RKNNVideoInfer:
    def __init__(self, rknn_path, input_size=(640, 640)):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        self.rknn.load_rknn(rknn_path)
        self.rknn.init_runtime()
        self.input_size = input_size

    def infer(self, frame):
        img = preprocess(frame, self.input_size)
        outputs = self.rknn.inference(inputs=[img])
        boxes, scores, classes = postprocess(outputs, frame.shape)
        return boxes, scores, classes

    def run_on_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            boxes, scores, classes = self.infer(frame)
            vis = draw_boxes(frame, boxes, scores, classes)
            cv2.imshow('RKNN Video Inference', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RKNN 视频推理可视化")
    parser.add_argument("rknn_path", type=str, help="RKNN 模型路径")
    parser.add_argument("video_path", type=str, help="视频文件路径")
    parser.add_argument("--input_size", type=int, nargs=2, default=[640, 640], help="输入尺寸")
    args = parser.parse_args()
    infer = RKNNVideoInfer(args.rknn_path, tuple(args.input_size))
    infer.run_on_video(args.video_path) 