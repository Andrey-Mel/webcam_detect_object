# import sys
import cv2
import threading
import queue
import numpy as np
# import os
import multiprocessing as mp
import torch
import time
import ultralytics
from ultralytics import YOLO
from queue import Empty



# !!! ОБЯЗАТЕЛЬНО
# if __name__ == '__main__':
#     mp.set_start_method('spawn', force=True)


# raw_frame_queue = queue.Queue(maxsize=30) # for record
# detection_input_queue = queue.Queue(maxsize=10) # for processing YOLO
# detection_output_queue = queue.Queue(maxsize=10) # processes frames with bb
# detection_output_queue2 = queue.Queue(maxsize=10)
# stop_event = threading.Event()


def camera_process(cam_id: int, dir_name: str , raw_frame_queue, detection_input_queue, 
                   detection_output_queue, stop_event):
    '''
        Процесс для одной камеры: захват, обработка, YOLO, запись
        Все очереди создаются из вне
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # инициализация модели пока сделано просто без проверок
    model = YOLO(r'C:\TASK_DETECT_DRONE\CODES_DETECT_DRONE\YOLO+calc_distance\weights_sdu_v3\best.pt')
   
    # flag stopped
    # stop_event = threading.Event()

    # fps cameras
    CAM_FPS = 30
    VIDEO_DIRATION = 30
    # VIDEO_FILENAME = rf"record_cam1\video_{time.strftime('%Y%m%d_%H%M%S')}.avi"


    # function for capture cadrs
    def capture_frames():
        print(f'Thread capture frame start... camera {cam_id} ')
        
        cap = cv2.VideoCapture(cam_id)    
        if cap.isOpened():
                
            while not stop_event.is_set():
                # print('In capture in loop while!!!!')
                ret, frame_in = cap.read()
                if not ret:
                    print("Not read frame")
                    break
                frame = cv2.resize(frame_in, (768, 1024), interpolation=cv2.INTER_AREA)   # ! 1024x768 уменьшаю потому что видео большое 360x640
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                            # !
                #send cadr to record
                if raw_frame_queue.full():
                    raw_frame_queue.get()  # delete from queue old frame
                # print(f'Put in detection input cam: {cam_id}')
                # assert isinstance(frame, np.ndarray), f'not frame in raw'
                raw_frame_queue.put(frame.copy()) # copy frame for record

                #send frame to the YOLO
                if detection_input_queue.full():
                    detection_input_queue.get()
                # print(f'Put in detection input cam: {cam_id}')
                # assert isinstance(frame, np.ndarray), f'not frame in input detect'
                detection_input_queue.put((cam_id, frame.copy()))                                   # add cam_id in input for detection
               
                # not load cpu
                time.sleep(1 / CAM_FPS)

        else:
            print(f'Error open camera idx: {cam_id}')
            stop_event.set()
        #     # return
        
        
        cap.release()
        stop_event.set()
        print(f'Thread capture frame STOP >>> camera {cam_id}')

    # function processing frame YOLO
    def process_frame():
        # print(f'Thread YOLO start... camera {cam_id}') 
        # if not stop_event.is_set():    
        while not stop_event.is_set(): 
            # print(f'In YOLO in loop while!!! {cam_id}')
            try:
                cam_idx, frame = detection_input_queue.get(timeout=1)
                if frame is None:
                    continue
                #processing frames YOLO
                result = model(frame, conf=0.4, iou=0.2, imgsz = 1024, device=device, verbose=False)
                annotated_frame = result[0].plot()
                # boxes = result[0].boxes.xyxy.cpu()
                # confs = result[0].bexes.conf.cpu()
                # idx = [int(i) for i in result[0].boxes.cls.cpu()]
                # for box, conf, id_cl in zip(boxes, confs, idx):
                #     x1 = box[0]
                #     y1 = box[1]
                #     x2 = box[2]
                #     y2 = box[3]

                #send a frame to show
                # if cam_idx == 0:
                if detection_output_queue.full():                        
                    detection_output_queue.get()                        
                detection_output_queue.put(annotated_frame)
                    # detection_input_queue.task_done()


            except queue.Empty:
                continue
            except Exception as e:
                if not stop_event.is_set():
                    print(f'CAM {cam_idx}, Error to in processing: {e}')
                # continue
        
        print(f'Thread YOLO stop! >>> camera {cam_idx}')

    # video recording funtion
    def record_video():
        print(f"Thread Write Video start... camera {cam_id}")
        frame_count = 0
        max_frame = VIDEO_DIRATION * CAM_FPS
        writer = None
        start_time = time.time()

        print(f"Record video start {VIDEO_DIRATION} sec")

        while not stop_event.is_set():
            # print('Record in loop while!!!!!')
            if frame_count < max_frame:  
                try:
                    frame = raw_frame_queue.get(timeout=1)
                    if writer is None:
                        h, w = frame.shape[:2]
                        VIDEO_FILENAME = rf"{dir_name}\video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # *'XVID'
                        writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, CAM_FPS, (w, h))

                    writer.write(frame)
                    frame_count += 1
                    # raw_frame_queue.task_done()

                    #show progress
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        print(f'Recorded {frame_count / max_frame} cadrs ({elapsed:.1f}) sec ')
                
                except queue.Empty:
                    continue


            else:
                writer.release()
                writer = None
                frame_count = 0
                print(f'Video record saves: {VIDEO_FILENAME}')
                
            
        if writer:
            writer.release()
            writer = None
            # stop_event.set()
                
        print(f'Thread write STOP>>> camera {cam_id}')


    # Cоздание и запись потоков
    # threads
    threads = [
        threading.Thread(target=capture_frames, name=f'Capture_{cam_id}'),
        threading.Thread(target=process_frame, name=f'Process YOLO_{cam_id}'),
        threading.Thread(target=record_video, name=f'Record_{cam_id}'),
       
    ]

    #start
    for t in threads:
        t.start()
        
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
            if stop_event.is_set():
                break   

    except KeyboardInterrupt:
        print("Stopped, was pressed Ctrl+C")
        stop_event.set()

    for t in threads:        
        t.join(timeout=3)    

    print(f'[CAM {cam_id}] All threads stoped!!!')



def display_img2img(detection_output_queue1, detection_output_queue2, stop_event):
    """
    Процесс отображения PiP: основное окно — камера 0, в углу — камера 1.
    """
    print('[DISPLAY] PiP отображение запущено...')
    pip_size = (200, 150)  # Размер вставки
    placeholder = None  # Заглушка для отсутствующего кадра

    while not stop_event.is_set():
        frame1 = None
        frame2 = None

        # Получаем кадр от камеры 0
        try:
            frame1 = detection_output_queue1.get(timeout=1)
        except Empty:
            print("[DISPLAY] Очередь камеры 0 пуста, пропуск кадра...")
            pass  # Оставляем frame1 = None
            

        # Получаем кадр от камеры 1
        try:
            frame2 = detection_output_queue2.get(timeout=1)
        except Empty:
            print("[DISPLAY] Очередь камеры 1 пуста, пропуск кадра...")
            pass  # Оставляем frame2 = None
            

        # Если хотя бы один кадр есть — отображаем
        if frame1 is not None:
            # Если нет frame2 — создаём placeholder
            if frame2 is None:
                if placeholder is None:
                    # Создаём серый placeholder такого же размера, как frame1 (или стандартного)
                    
                    placeholder = np.full((pip_size[1], pip_size[0], 3), 128, dtype=np.uint8)  # серый
                    cv2.putText(placeholder, "NO SIGNAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                frame2_resized = placeholder
            else:
                frame2_resized = cv2.resize(frame2, pip_size)

            # Наложение PiP
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2_resized.shape[:2]

            x_offset = w1 - w2 - 10
            y_offset = 10

            # Проверка границ
            if y_offset + h2 <= h1 and x_offset + w2 <= w1:
                frame1[y_offset:y_offset+h2, x_offset:x_offset+w2] = frame2_resized
            else:
                print("[DISPLAY] Не удалось наложить PiP — размеры не подходят.")
            # frame_rz = cv2.resize(frame1, (480, 640))
            cv2.imshow('PiP View (Cam0 + Cam1)', frame1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        else:
            print("[DISPLAY] Нет кадра от камеры 0 — пропуск отображения.")
            time.sleep(0.1)  # Не грузить CPU

    cv2.destroyAllWindows()
    print('[DISPLAY] Отображение остановлено.')
    

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # main()
    sorce_vid = r'E:\DB_SDU\test_video\fire2_test.mp4' #r'C:\TASK_DETECT_DRONE\CODES_DETECT_DRONE\YOLO+calc_distance\record_cam1\video_1024_768.mp4' #
    print(f'CPU count: {mp.cpu_count()}')

    # Создаем очереди и событие остановки
    raw_frame_queue1 = mp.Queue(maxsize=30)
    raw_frame_queue2 = mp.Queue(maxsize=30)
    detection_input_queue1 = mp.Queue(maxsize=20) # общая очередь для обеих камер
    detection_input_queue2 = mp.Queue(maxsize=20)
    detection_output_queue1 = mp.Queue(maxsize=10) # for cam 1
    detection_output_queue2 = mp.Queue(maxsize=10) # for cam 2
    stop_event = mp.Event()
  

    processes = [
        mp.Process(target=camera_process, 
                   args=(0, 'record_cam1', raw_frame_queue1, detection_input_queue1, 
                         detection_output_queue1, stop_event),name='Camera 0'),
        mp.Process(target=camera_process, 
                   args=(1, 'record_cam2', raw_frame_queue2, detection_input_queue2, 
                        detection_output_queue2, stop_event),name='Camera 1'),
        mp.Process(target=display_img2img, args=(detection_output_queue1, detection_output_queue2, stop_event), name='Dispay')
    ]

    for p in processes:
        print(f'Name process: {p.name}')
        p.start()
    
    print(f'Start method: {mp.get_start_method()}')
    
    try: # main loop , wait press Ctrl + C
        while any(p.is_alive() for p in processes):
            time.sleep(1)
           
    except KeyboardInterrupt:
        print('\nStopped all process...')
        stop_event.set()

    print(f'Active children: {mp.active_children()}')

    # Stopped acuratno all process
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            print(f'[MAIN] Принудительное завершение {p.name}')
            p.terminate() # send SIGTERM
            p.join()
            
        print(f'Exitcode: {p.exitcode}')
    
    print('ALL PROCESSES CAMERAS STOPPED!!!')

