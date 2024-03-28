
import argparse
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import pyzed.sl as sl
from threading import Lock, Thread
from time import sleep

lock = Lock()
run_signal = False
exit_signal = False

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

def torch_thread(weights, img_size, conf_thres=0.4, iou_thres=0.5):
    global image_net, exit_signal, run_signal, detections, point_cloud, image_left

    print("Intializing Network...")
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA
    imgsz = img_size

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


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
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            '''pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)'''

            for i, det in enumerate(pred):
                s, im0 = '', image_net.copy()
                gn = torch.tensor(image_net.shape)[[1, 0, 1, 0]]
                names = model.module.names if hasattr(model, 'module') else model.names
                annotator = Annotator(im0, line_width=3, example=str(names))


                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    cent_x = round(xywh[0] * im0.shape[1])
                    cent_y = round(xywh[1] * im0.shape[0])
                    cent_w = round(xywh[2] * im0.shape[1])
                    point_1 = round(cent_x - 0.4 * cent_w)
                    point_2 = round(cent_x + 0.4 * cent_w)
                    wide_value_1 = point_cloud.get_value(point_1, cent_y)[1]
                    wide_value_2 = point_cloud.get_value(point_2, cent_y)[1]

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
                            annotator.box_label(xyxy, label, color=colors(c, True))

                        except:
                            pass
                    # im = annotator.result()
                im0 = annotator.result()
                cv2.imshow('00', im0)
                key = cv2.waitKey(1)
                if key == 'q':
                    break

            lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections, point_cloud, image_left

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
    parser.add_argument('--weights', nargs='+', type=str, default='gelan-c-det.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    opt = parser.parse_args()
    with torch.no_grad():
        main()
