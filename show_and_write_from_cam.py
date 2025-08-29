import sys
import cv2
import threading
import queue
import os
import multiprocessing as mp
import torch
import time
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# инициализация модели пока сделано просто без проверок
model = YOLO('yolo11m.pt')

# print(f'{sys.executable}')
# # print(f'{sys.path}')

# очереди
raw_frame_queue = queue.Queue(maxsize=30) # for record
detection_input_queue = queue.Queue 





