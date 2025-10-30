# код из репозитория https://github.com/airockchip/rknn-toolkit2
# Это пример запуска из симулятора, но он тоже не выдаетс детекцию
import cv2
import numpy as np
# from rknnlite.api import RKNNLite
from rknn.api import RKNN
import time
import argparse

RKNN_MODEL = 'best_sdu_v3.rknn'
ONNX_MODEL = 'best_sdu_v3.onnx'
IMGSZ = (640, 640)
IMG = '/mnt/e/DB_SDU/ds_for_rknn/animal.jpg' #animal.jpg  animal1_rsz
IMG_ORIG = '/mnt/e/DB_SDU/ds_for_rknn/animal_orig.jpg' #animal_orig.jpg'

CLASSES = ("person", "car", 'fire', 'animal')
# CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
#            "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
#            "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
#            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
#            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
#            "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
#            "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im

def preprocess(img_path):
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, IMGSZ)
    img = letterbox(img)
    
    img = np.expand_dims(img, axis=0)
    img = img.transpose((0, 3, 1, 2))
    print('IMG SHAPE: ', img.shape)
    return img
# def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
#     """
#     Простая реализация NMS на NumPy.
#     boxes: (N, 4) в формате [x1, y1, x2, y2]
#     scores: (N,)
#     """
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]

#     areas = (x2 - x1) * (y2 - y1)
#     order = scores.argsort()[::-1]  # сортируем по убыванию

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)

#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0.0, xx2 - xx1)
#         h = np.maximum(0.0, yy2 - yy1)
#         inter = w * h

#         iou = inter / (areas[i] + areas[order[1:]] - inter)
#         inds = np.where(iou <= iou_threshold)[0]
#         order = order[inds + 1]

#     return np.array(keep)

def postprocess(output, confidence_thres=0.3, iou_thres=0.5):
    outputs = np.transpose(np.squeeze(output[0]))
    
    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    x_factor = 1
    y_factor = 1

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        # print(outputs[i][:8])
        classes_scores = outputs[i][4:]
    
        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        # if max_score >= confidence_thres:
            # Get the class ID with the highest score
        class_id = np.argmax(classes_scores)

        # Extract the bounding box coordinates from the current row
        x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

        # Calculate the scaled coordinates of the bounding box
        x1 = int((x - w / 2) * x_factor)
        y1 = int((y - h / 2) * y_factor)
        x2 = x1 + int(w * x_factor)
        y2 = y1 + int(h * y_factor)
        
        
        # Add the class ID, score, and box coordinates to the respective lists
        class_ids.append(class_id)
        scores.append(max_score)
        boxes.append([x1, y1, x2, y2])
    # print('POSTPROCESS: COORD ', boxes, '\n', \
    #     'POSTPROCESS: CL_ID - SCORE ', class_ids, ' - ', scores)
    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    # indices = nms(boxes, scores, iou_thres)
    print('INDICES: ', indices)
    detections = []

    # Iterate over the selected indices after non-maximum suppression
    for i in indices:
        detections.append([
            boxes[i],
            scores[i],
            class_ids[i]
        ])

    # Return the modified input image
    return detections

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--img', type=str, default='bus.jpg')
    # parser.add_argument('--model', type=str, default='yolov8m_RK3588_i8.rknn')
    # opt = parser.parse_args()
    # args = vars(opt)
    # rknn_model = 'yolo11m.rknn'
    rknn = RKNN()

    #config
    print('--> config model')
    # rknn.config(target_platform='rk3588')
    rknn.config(mean_values=[[0, 0, 0]], 
                std_values=[[255, 255, 255]], 
                quant_img_RGB2BGR = True,
                target_platform='rk3588',
                quantized_algorithm='normal', #'kl_divergence',
                quantized_method='channel',
                optimization_level = 3,
                quantized_dtype = 'asymmetric_quantized-8' #'w8a8'
                )
    print('done')
    print('DONE config')
    
    print('--> Load ONNX model')
    ret = rknn.load_onnx(ONNX_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret) 
    print('Done load ONNX')

    print('BUILD model')
    ret = rknn.build(do_quantization=True, dataset='dataset.txt')
    if ret != 0:
        print('Build failed model')
        exit(ret)
    print('Done BUILD model')

    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed')
        exit(ret)
    print('Done EXPORT rknn model')

    
    print('--> Init runtime enviroment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)


    print('RUNNING MODEL')
    
    img_data = preprocess(IMG_ORIG)
    start = time.time()
    outputs = rknn.inference(inputs=[img_data], data_format='nchw')
    print(f"inference time: {(time.time() - start) * 1000} ms")  #, data_format='nchw'
    print('OUTPUTS: ', outputs[0].shape, '\nmin: ', outputs[0].min(), '\nmax: ', outputs[0].max())
    detections = postprocess(outputs[0])
    print(f"detection time: {(time.time() - start) * 1000} ms")
    print('DETECTIONS: ', len(detections), '\n', detections)

    img_orig = cv2.imread(IMG)
    # img_orig = cv2.resize(img_orig, IMGSZ)

    for d in detections:
        score, class_id = d[1], d[2]
        x1, y1, x2, y2 = d[0][0], d[0][1], d[0][2], d[0][3]
        cv2.rectangle(img_orig, (x1, y1), (x2, y2), 2)
        label = f'{CLASSES[class_id]}: {score:.2f}'
        label_height = 10
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.putText(img_orig, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite('SDU_result.jpg', img_orig)

    print(f"{(time.time() - start) * 1000} ms")
    rknn.release()