# import sys
import cv2
import threading
import queue
# import os
import multiprocessing as mp
import torch
import time
import ultralytics
from ultralytics import YOLO


# !!! ОБЯЗАТЕЛЬНО
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)


def camera_process(cam_id: int = 0, dir_name: str = 'record_cam1'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # инициализация модели пока сделано просто без проверок
    model = YOLO('yolo11m.pt')
    # print(type(model)) # 'ultralytics.models.yolo.model.YOLO'

    # print(f'{sys.executable}')
    # # print(f'{sys.path}')

    # очереди  (queues)
    raw_frame_queue = queue.Queue(maxsize=30) # for record
    detection_input_queue = queue.Queue(maxsize=10) # for processing YOLO
    detection_output_queue = queue.Queue(maxsize=10) # processes frames with bb

    # flag stopped
    stop_event = threading.Event()

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
                ret, frame = cap.read()
                if not ret:
                    print("Not read frame")
                    break

                #send cadr to record
                if raw_frame_queue.full():
                    raw_frame_queue.get()  # delete from queue old frame
                
                raw_frame_queue.put(frame.copy()) # copy frame 

                #send frame to the YOLO
                if detection_input_queue.full():
                    detection_input_queue.get()
                detection_input_queue.put(frame.copy())

                # not load cpu
                time.sleep(1 / CAM_FPS)

        else:
            print(f'Error open camera idx: {cam_id}')
            stop_event.set()
            # return
        
        
        cap.release()
        print(f'Thread capture frame STOP >>> camera {cam_id}')

    # function processing frame YOLO
    def process_frame():
        # print(f'Thread YOLO start... camera {cam_id}') 
        if not stop_event.is_set():    
            while not stop_event.is_set(): 
                # print(f'In YOLO in loop while!!! {cam_id}')
                try:
                    frame = detection_input_queue.get()
                    if frame is None:
                        continue
                    #processing frames YOLO
                    result = model(frame, conf=0.5, iou=0.2, imgsz=640, device=device, verbose=False)
                    annotated_frame = result[0].plot()

                    #send a frame to show
                    if detection_output_queue.full():
                        detection_output_queue.get()
                    detection_output_queue.put(annotated_frame)

                    detection_input_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f'Error to in processing: {e}')
        # else:
        #     print('Block ELSE')
        #     stop_event.set()

        # stop_event.set()
                
        print(f'Thread YOLO stop! >>> camera {cam_id}')

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
                    raw_frame_queue.task_done()

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
            stop_event.set()
                
        print(f'Thread write STOP>>> camera {cam_id}')

    # display frames fuction
    def display_frames():
        print(f"Thread display cadrs start... camera {cam_id}")
        while not stop_event.is_set():
            # print('In Display in loop!!!!!')
            try:
                frame = detection_output_queue.get(timeout=1)
                cv2.imshow(f'YOLO detection {cam_id}', frame)
                detection_output_queue.task_done()
            except queue.Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        cv2.destroyAllWindows()
        print(f'Thread display stop>>> camera {cam_id}')

    # MAIN, start of all function on the thread
    # def main():
    print("Start system threading, for stop press 'q'...")

    # threads
    threads = [
        threading.Thread(target=capture_frames, name='Capture'),
        threading.Thread(target=process_frame, name='Process YOLO'),
        threading.Thread(target=record_video, name='Record'),
        threading.Thread(target=display_frames, name='Display')
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

    print('All threads stoped!!!')


if __name__ == '__main__':
    # main()
    print(f'CPU count: {mp.cpu_count()}')
    
    processes = [
        mp.Process(target=camera_process, args=(0, 'record_cam1'), daemon=False),
        mp.Process(target=camera_process, args=(1, 'record_cam2'), daemon=False)
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

    print(f'Active children: {mp.active_children()}')

    # Stopped acuratno all process
    for p in processes:
        p.terminate() # send SIGTERM
        p.join(timeout=2)
        print(f'Exitcode: {p.exitcode}')
    
    print('ALL PROCESSES CAMERAS STOPPED!!!')

