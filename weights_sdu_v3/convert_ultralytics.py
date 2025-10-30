# Этот код запускается из под wsl в виндовс, сама модель запускается на radxa, в wsl модель тоже не запускается,
# так как она для платы конвертируется
import numpy as np
import os
import time
from ultralytics import YOLO


# model = YOLO(r'best_sdu_v3.pt')
model = YOLO('best_sdu_v3.pt')


model.export(format='rknn', name='rk3588', imgsz=[640, 640]) #, batch=1, nms=False, simplify=True, opset=13

# print('Test for inference')
# time.sleep(5)
# # try:
# print("Tried inferenc on comp")
# model_r = YOLO('./best_sdu_v3_rknn_model')

# result = model_r('0000069_00713_d_0000003.jpg')
# result[0].show()

# except Exception as e:
#     print(f'Error conver model: {e}')
