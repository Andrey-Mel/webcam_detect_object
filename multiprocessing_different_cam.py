import cv2
import multiprocessing as mp
import time


def capture_camera_process(idx_cam: int, queue: mp.Queue):
    """function to work in diferent process - capture frame and sent in queue"""
    cap = cv2.VideoCapture(idx_cam)
    if not cap.isOpened():
        print(f'Cam {idx_cam} not opened')
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Процесс {idx_cam}] Подключено: {width}x{height} @ {fps}fps")


    while True:
        ret, frame = cap.read()
        if not ret:
            print(f'Not get frame')
            break
        # Добавляем текст на кадре (можно и в главном процессе, но так легче)
        text = f'Cam {idx_cam} | {width}x{height} @ {fps}fps'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # if queue is full
        if not queue.empty():
            queue.get() # cleareing queue
        queue.put((idx_cam, frame))

        
        time.sleep(0.01)
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
    try:
        while True:
            frame1 = None
            frame2 = None

            if not queue1.empty():
                cam_id, frame1 = queue1.get()
                
                if frame1 is None:
                    print('Camera 1 is stop')
                    break
                cv2.imshow(f'Camera_{cam_id}', frame1)
            
            if not queue2.empty():
                cam_id, frame2 = queue2.get()
                if frame2 is None:
                    print('Camera 2 is stop')
                    break
                cv2.imshow(f'Camera {cam_id}', frame2)

            # if frame1 is not None:
            #     cv2.imshow(f'Camera {cam_id}', frame2)
            # if frame2 is not None:
            #     cv2.imshow(f'Camera {cam_id}', frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Exit by command user')
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print('Stopped USER')

    finally:
        # stoped processes
        cv2.destroyAllWindows()
        proc1.terminate()
        proc2.terminate()
        proc1.join(timeout=1)
        proc2.join(timeout=1)
        cv2.destroyAllWindows()
        print('Resourses to free')

