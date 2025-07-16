import cv2
import numpy as np

def preprocess(frame, input_size):
    # resize + BGR2RGB + HWC2CHW + 归一化
    img = cv2.resize(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0)  # NCHW
    return img

def postprocess(outputs, orig_shape, conf_thres=0.3, iou_thres=0.5):
    # 假设 outputs[0] 为 [num, 6]，格式: [x1, y1, x2, y2, conf, cls]
    preds = outputs[0]
    boxes, scores, classes = [], [], []
    for pred in preds:
        x1, y1, x2, y2, conf, cls = pred
        if conf < conf_thres:
            continue
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(float(conf))
        classes.append(int(cls))
    return boxes, scores, classes

def draw_boxes(frame, boxes, scores, classes, class_names=None):
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        color = (0, 255, 0)
        label = f"{cls}:{score:.2f}" if class_names is None else f"{class_names[cls]}:{score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame 