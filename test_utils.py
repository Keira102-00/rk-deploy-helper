import cv2
import numpy as np
from inference.utils import preprocess, postprocess, draw_boxes

# 生成一张随机图片
img = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)

# 测试 preprocess
input_tensor = preprocess(img, (640, 640))
print('preprocess输出shape:', input_tensor.shape)

# 构造模拟模型输出，格式: [x1, y1, x2, y2, conf, cls]
preds = np.array([
    [100, 120, 300, 350, 0.85, 0],
    [200, 220, 400, 450, 0.65, 1],
])
outputs = [preds]
boxes, scores, classes = postprocess(outputs, img.shape)
print('postprocess输出:', boxes, scores, classes)

# 测试 draw_boxes
img_with_boxes = draw_boxes(img.copy(), boxes, scores, classes)
cv2.imwrite('test_utils_result.jpg', img_with_boxes)
print('已保存可视化结果 test_utils_result.jpg') 