import cv2
import multiprocessing as mp
import time
import torch
from ultralytics import YOLO


device = 'cuda' if torch.cuda.is_available() else 'cpu'

writers = {}

model = YOLO('yolo11m.pt')

def capture_camera_process(idx_cam: int, queue: mp.Queue):
    """function to work in diferent process - capture frame and sent in queue"""
    cap = cv2.VideoCapture(idx_cam)#, cv2.CAP_DSHOW
    if not cap.isOpened():
        print(f'Cam {idx_cam} not opened')
        return

    #set params for cameras fps, width, height
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)


    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # if fps == 0:
    #     fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Процесс {idx_cam}] Подключено: {width}x{height} @ {fps}fps")

   
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f'Not get frame')
            break
        result = model(frame, conf=0.5, iou=0.2, imgsz=1024, verbose=False, device=device)
        
        cls = result[0].boxes.cls.cpu().numpy()
        coords = result[0].boxes.xyxy.cpu().numpy()
        for cl, (x, y, x1, y1) in zip(cls, coords):
            cv2.rectangle(frame, (int(x), int(y)), (int(x1), int(y1)), (255, 0, 0), 1)
            # Добавляем текст на кадре (можно и в главном процессе, но так легче)
            
            text_cls = str(cl)
            cv2.putText(frame, text_cls, (int(x), int(y-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),1)
        text = f'Cam {idx_cam} | {width}x{height} @ {fps}fps'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # if queue is full
        if not queue.empty():
            queue.get() # cleareing queue
        queue.put((idx_cam, frame, fps, width, height))

        
        time.sleep(0.08)
        
    cap.release()
    queue.put((idx_cam, None)) # Signal to end
    print(f'Process cam {idx_cam} END.')


if __name__=='__main__':
    mp.set_start_method('spawn', force=True)  # Критично для Windows/macOS
    idx_cam_1 = 0
    idx_cam_2 = 1

    # Создаём очереди для передачи кадров из процессов
    queue1 = mp.Queue(maxsize=1)
    queue2 = mp.Queue(maxsize=1)

    # Запускаем процессы
    proc1 = mp.Process(target=capture_camera_process, args=(idx_cam_1, queue1), daemon=True)
    proc2 = mp.Process(target=capture_camera_process, args=(idx_cam_2, queue2), daemon=True)

    proc1.start()
    proc2.start()

    print('Start processes to capture cadrs. Press "q" to exit')

    current_time = time.strftime("%Y%m%d_%H%M%S")
    
    try:
        while True:
            
            frame1 = None
            frame2 = None

            if not queue1.empty():
                cam_id, frame1, fps, width, height = queue1.get()
                
                if frame1 is None:
                    print('Camera 1 is stop')
                    break
                cv2.imshow(f'Camera_{cam_id}', frame1)
                # writer 1
                if cam_id not in writers:
                    filename1 = fr"record_cam1\camera_{cam_id}_{current_time}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V') # or 'MP4V' for mp4, 'xvid' for avi
                    writers[cam_id] = cv2.VideoWriter(filename1, fourcc, fps, (width, height))
                    print(f'Record start from cam - {cam_id}')
                writers[cam_id].write(frame1)
                
            
            if not queue2.empty():
                cam_id, frame2, fps, width, height = queue2.get()
                if frame2 is None:
                    print('Camera 2 is stop')
                    break
                cv2.imshow(f'Camera {cam_id}', frame2)
                if cam_id not in writers:
                    filename2 = fr"record_cam2\camera_{cam_id}_{current_time}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V') # or 'MP4V' for mp4, xvid for avi
                    writers[cam_id] = cv2.VideoWriter(filename2, fourcc, fps, (width, height))
                    print(f'Record start from cam - {cam_id}')
                writers[cam_id].write(frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Exit by command user')
                break
            time.sleep(0.01)
            cur_time = time.time()
            
    except KeyboardInterrupt:
        print('Stopped USER')

    finally:
        # stoped processes
        for writer in writers.values():
            writer.release()
        cv2.destroyAllWindows()
        
        proc1.terminate()
        proc2.terminate()
        proc1.join(timeout=1)
        proc2.join(timeout=1)
        
        print('Resourses to free')

