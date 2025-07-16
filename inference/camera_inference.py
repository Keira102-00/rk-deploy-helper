import cv2
import numpy as np
from utils import preprocess, postprocess, draw_boxes

class RKNNCameraInfer:
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

    def run_on_camera(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            boxes, scores, classes = self.infer(frame)
            vis = draw_boxes(frame, boxes, scores, classes)
            cv2.imshow('RKNN Camera Inference', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RKNN 摄像头推理可视化")
    parser.add_argument("rknn_path", type=str, help="RKNN 模型路径")
    parser.add_argument("--camera_id", type=int, default=0, help="摄像头ID")
    parser.add_argument("--input_size", type=int, nargs=2, default=[640, 640], help="输入尺寸")
    args = parser.parse_args()
    infer = RKNNCameraInfer(args.rknn_path, tuple(args.input_size))
    infer.run_on_camera(args.camera_id) 