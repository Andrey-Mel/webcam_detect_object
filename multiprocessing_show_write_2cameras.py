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
    model = YOLO(r'C:\TASK_DETECT_DRONE\CODES_DETECT_DRONE\YOLO+calc_distance\weights_sdu_v3\best_sdu_v3.pt') #YOLO('yolo11m.pt')
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
                    stop_event.set()
                    break
                frame = cv2.resize(frame, (1024, 768), interpolation=cv2.INTER_AREA) # 640, 420, 1024 - 768
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

        # added later self write befor git                                      !!!!!
        def plot(reslt, arr_img):
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (100, 100, 100)]
            class_names = reslt[0].names
            names_clases = [class_names[int(i)] for i in reslt[0].boxes.cls.cpu()]
            boxes = reslt[0].boxes.xyxy.cpu()
            confidences = reslt[0].boxes.conf.cpu()
            color_clases = dict(zip(class_names.values(), color))
            bad_w, bad_h = 7, 7

            for box, conf, name in zip(boxes, confidences, names_clases):
                
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                if (x2 - x1) <= bad_w and (y2 - y1) <= bad_h: 
                    continue
                conf = round(float(conf), 2)
                color_ = color_clases.get(name, None)
                cv2.rectangle(arr_img, (x1, y1), (x2, y2), color_, 1)
                cv2.putText(arr_img, f'{name}_{conf}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_, 1)
            return arr_img
        
        def plot_tracking(res, img_arr):
            if res[0].boxes and res[0].boxes.is_track:
               
                boxes = res[0].boxes.xyxy.cpu()
                track_ids = res[0].boxes.id.int().cpu()
                confidences = res[0].boxes.conf.cpu()
                clses = res[0].boxes.cls.cpu()
                bad_w, bad_h = 7, 7
                for box, track_id, conf, cl in zip(boxes, track_ids, confidences, clses):
                    x1, y1, x2, y2 = map(int, box)
                    if (x2 - x1) <= bad_w or (y2 - y1) <= bad_h: 
                        continue
                    conf = round(float(conf), 2)
                    cv2.rectangle(img_arr, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(img_arr, f'{track_id}_{cl}_{conf}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                return img_arr
            else:
                return img_arr


        if not stop_event.is_set():    
            while not stop_event.is_set(): 
                # print(f'In YOLO in loop while!!! {cam_id}')
                try:
                    frame = detection_input_queue.get()
                    if frame is None:
                        continue
                    start_tm = time.time()
                    #processing frames YOLO
                    # without tracking
                    result = model(frame, conf=0.4, iou=0.3, imgsz=1024, device='cpu', verbose=False)

                    # with tracking
                    # result = model.track(
                    #     frame,
                    #     imgsz = 1024,
                    #     conf = 0.35,
                    #     iou = 0.2,
                    #     device=device,
                    #     persist=False,
                    #     verbose = False,
                    #     tracker = 'conf_trackers/botsort_custom.yaml',
                        
                    # )
                    
                    annotated_frame = result[0].plot()
                    # annotated_frame = plot_tracking(result, frame)
                    
                    print(f'INFERENCE AND POSTPROC 1 FRAME: {round(time.time() - start_tm, 2)}')
                    

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
    
    # Add timing                                                        !!! 
    start_time = time.time()
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
    print(f'END DETECT TIME: {round(time.time() - start_time, 2)}')
    print('All threads stoped!!!')


if __name__ == '__main__':
    # torch.cuda.reset_max_memory_allocated()
    # torch.cuda.reset_max_memory_allocated()
    # main()
    print(f'CPU count: {mp.cpu_count()}')
    sorce_video = r'E:\DB_SDU\test_video\1024_768_cow_.mp4'  # kosuli kozy2
    processes = [
        mp.Process(target=camera_process, args=(sorce_video, 'record_cam1'), daemon=False),
        # mp.Process(target=camera_process, args=(1, 'record_cam2'), daemon=False)
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

