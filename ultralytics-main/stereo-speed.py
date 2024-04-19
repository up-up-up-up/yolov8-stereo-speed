import time
import cv2
import numpy as np

from ultralytics import YOLO
from stereo import stereoconfig_040_2
from stereo.stereo import stereo_40
from stereo.stereo import stereo_threading, MyThread
from stereo.dianyuntu_yolo import preprocess, undistortion, getRectifyTransform, draw_line, rectifyImage, \
    stereoMatchSGBM
from collections import defaultdict

def detect():

    model = YOLO("yolov8n.pt")
    cv2.namedWindow('Speed Estimation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Speed Estimation', 1280, 360)  # 设置宽高
    cap = cv2.VideoCapture('ultralytics/assets/a2.mp4')
    out_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (2560, 720))
    previous_positions = {}
    track_history = defaultdict(lambda: [])
    assert cap.isOpened(), "Error reading video file"
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        # start_time = time.time()
        im0 = cv2.resize(im0, (2560, 720))
        config = stereoconfig_040_2.stereoCamera()
        map1x, map1y, map2x, map2y, Q = getRectifyTransform(720, 1280, config)
        thread = MyThread(stereo_threading, args=(config, im0, map1x, map1y, map2x, map2y, Q))
        thread.start()
        start_time = time.time()
        tracks = model.track(im0, persist=True)
        track_ids = tracks[0].boxes.id.int().cpu().tolist()
        annotated_frame = tracks[0].plot()
        boxes = tracks[0].boxes.xywh.cpu()

        # for i, box in enumerate(boxes):
        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):

            x_center, y_center, width, height = box.tolist()
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            if (0 < x2 < 1280):
                thread.join()
                points_3d = thread.get_result()
                a = points_3d[int(y_center), int(x_center), 0] / 1000
                b = points_3d[int(y_center), int(x_center), 1] / 1000
                c = points_3d[int(y_center), int(x_center), 2] / 1000
                distance = ((a ** 2 + b ** 2 + c ** 2) ** 0.5)
                bbox_center_3d = (float(a), float(b), float(c))
                current_position = (a, b, c)
                if i not in previous_positions:
                    previous_positions[i] = current_position
                    continue
                tracks.append(bbox_center_3d)
                previous_x, previous_y, previous_z = previous_positions[i]
                displacement = ((a - previous_x) ** 2 + (b - previous_y) ** 2 + + (c - previous_z) ** 2) ** 0.5
                end_time = time.time()
                run_time = end_time-start_time
                speed = displacement / run_time
                print(run_time)
                # 更新上一帧位置
                previous_positions[i] = current_position
                # 在这里使用速度进行你想要的操作，比如打印出来
                # print("目标 {} 的速度为: {} m/s".format(i, speed))
                print("速度为: {} m/s".format(speed))

                track = track_history[track_id]
                track.append((float(x_center), float(y_center)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=3)

                if (distance != 0):
                    text_dis_avg = "dis:%0.2fm" % distance
                    cv2.putText(annotated_frame, text_dis_avg, (int(x2 + 5), int(y1 + 30)), cv2.FONT_ITALIC, 1.0,
                                (0, 255, 255), 2)
                    text_speed = "v:%0.2fm/s" % speed
                    cv2.putText(annotated_frame, text_speed, (int(x2 + 5), int(y1 + 60)), cv2.FONT_ITALIC, 1.0,
                                (255, 255, 0), 2)

        cv2.imshow("Speed Estimation", annotated_frame)
        out_video.write(annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    out_video.release()
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    detect()