import argparse
import time
from pathlib import Path
from threading import Lock, Thread
from time import sleep
import cv2
import torch
from utils.datasets import LoadStreams, LoadImages
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
# from utils.augmentations import letterbox
from utils.plots import plot_one_box
from threading import Lock, Thread
from time import sleep
import torch.backends.cudnn as cudnn
from numpy import random
import pyzed.sl as sl
import threading
from utils.utils import *

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from threading import Lock, Thread
from time import sleep

lock = Lock()
run_signal = False
exit_signal = False
from stereo.dianyuntu_yolo import preprocess, undistortion, getRectifyTransform, draw_line, rectifyImage, \
    stereoMatchSGBM  # , hw3ToN3, view_cloud ,DepthColor2Cloud
from stereo import stereoconfig_040_2
from stereo.stereo import get_median, stereo_threading, MyThread

intermediate = 0
high = 0
normal = 0
people_label = " "
normal_label = " "
inter_label = " "
high_label = " "

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5 * xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5 * xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5 * xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5 * xywh[3]) * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max
    return output
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def torch_thread(weights, img_size, conf_thres=0.6, iou_thres=0.9):
    global image_net, exit_signal, run_signal, detections, point_cloud

    print("Intializing Network...")
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = img_size


    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if trace:
    # model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16
    cudnn.benchmark = True

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    t0 = time.time()

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    while not exit_signal:
        if run_signal:
            lock.acquire()

            img, ratio, pad = letterbox(image_net[:, :, :3], imgsz, auto=False)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img = img / 255.0
            if len(img.shape) == 3:
                img = img[None]
            '''for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)'''
            #############################################
            '''pred = model(img, augment=False, visualize=False)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)'''
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

            for i, det in enumerate(pred):
                s, im0 = '', image_net.copy()
                gn = torch.tensor(image_net.shape)[[1, 0, 1, 0]]
                # annotator = Annotator(image_net, line_width=2, example=str('A'))

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_net.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        cent_x = round(xywh[0] * im0.shape[1])
                        cent_y = round(xywh[1] * im0.shape[0])
                        cent_w = round(xywh[2] * im0.shape[1])
                        point_1 = round(cent_x - 0.4 * cent_w)
                        point_2 = round(cent_x + 0.4 * cent_w)
                        wide_value_1 = point_cloud.get_value(point_1, cent_y)[1]
                        wide_value_2 = point_cloud.get_value(point_2, cent_y)[1]

                        x1 = xyxy[0]
                        x2 = xyxy[2]
                        y1 = xyxy[1]
                        y2 = xyxy[3]
                        try:
                            wide = round(wide_value_1[0], 4) - round(wide_value_2[0], 4)
                            wide = round(abs(wide * 1000))
                        except:
                            wide = 0.00
                            pass
                        point_cloud_value = point_cloud.get_value(cent_x, cent_y)[1]
                        point_cloud_value = point_cloud_value * -1000.00
                        if point_cloud_value[2] > 0.00:
                            try:
                                point_cloud_value[0] = round(point_cloud_value[0])
                                point_cloud_value[1] = round(point_cloud_value[1])
                                point_cloud_value[2] = round(point_cloud_value[2])
                                print("x:", point_cloud_value[0], "y:", point_cloud_value[1], "z:",
                                      point_cloud_value[2], "W:", wide)
                                txt = 'x:{0} y:{1} z:{2} w:{3}'.format(point_cloud_value[0], point_cloud_value[1],
                                                                       point_cloud_value[2], wide)
                                a=point_cloud_value[0]
                                b=point_cloud_value[1]
                                c=point_cloud_value[2]
                                distance = ((a ** 2 + b ** 2 + c ** 2) ** 0.5)

                                # annotator.box_label(xyxy, txt, color=(255, 0, 0))
                                label = f'{names[int(cls)]} {conf:.2f} '
                                label = label + " " +"dis:" +str(distance)

                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                            except:
                                pass
                        # im = annotator.result()
                        cv2.imshow('00', im0)
                        key = cv2.waitKey(10)
                        if key == 'q':
                            break

            lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections, point_cloud

    capture_thread = Thread(target=torch_thread,
                            kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 5

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()

    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    point_cloud_render = sl.Mat()

    point_cloud = sl.Mat()
    image_left = sl.Mat()
    depth = sl.Mat()
    # Utilities for 2D display

    while True and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)
            zed.retrieve_image(image_left, sl.VIEW.LEFT)


        else:
            exit_signal = True
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    #parser.add_argument('--source', type=str, default='a5.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    with torch.no_grad():
        main()
