import csv
import sys

import cv2
import time
import math
import torch
import rospy
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from utils.timer import Timer
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm


parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='FDDB', type=str, choices=['FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.9, type=float, help='visualization_threshold')
args = parser.parse_args()


class ros_fddb:
    def __init__(self):
        rospy.init_node('face_detector', anonymous=True)
        self.img_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.img_callback)
        self.pub = rospy.Publisher('/face_mark', Image, queue_size=10)

        self.face_mark_msg = Image()
        self.bridge = CvBridge()

        torch.set_grad_enabled(False)
        self.cfg = None
        self.device = None

        if args.network == "mobile0.25":
            self.cfg = cfg_mnet

        elif args.network == "resnet50":
            self.cfg = cfg_re50

        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = self.load_model(self.net, args.trained_model, args.cpu)
        self.net.eval()
        # print('Finished loading model!')

        cudnn.benchmark = True
        self.device = torch.device("cpu" if args.cpu else "cuda")
        self.net = self.net.to(self.device)

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            rate.sleep()


    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # print('Missing keys:{}'.format(len(missing_keys)))
        # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        # print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        #print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model(self, model, pretrained_path, load_to_cpu):
        #print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')

        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model


    def img_callback(self, data):
        if self.device:
            start = time.time()
            resize = 1

            _t = {'forward_pass': Timer(), 'misc': Timer()}

            img_raw = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            img = np.float32(img_raw)

            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = scale.to(self.device)

            _t['forward_pass'].tic()
            loc, conf, landms = self.net(img)  # forward pass
            _t['forward_pass'].toc()
            _t['misc'].tic()
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])

            scale1 = scale1.to(self.device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            #order = scores.argsort()[::-1][:args.top_k]
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)

            dets = dets[keep, :]
            landms = landms[keep]


            dets = np.concatenate((dets, landms), axis=1)
            _t['misc'].toc()

            for b in dets:
                if b[4] < args.vis_thres:
                    continue

                # pixel RGB 값 출력
                landmark_names = ["RIGHT Eye", "LEFT Eye", "Nose", "RIGHT Mouth", "LEFT Mouth"]

                print("Pixel RGB Information")
                print("====================================")
                for i, name in enumerate(landmark_names):
                    x = int(b[5 + i * 2])
                    y = int(b[6 + i * 2])

                    # 좌표가 이미지 범위 내에 있는지 확인
                    if 0 <= x < img_raw.shape[1] and 0 <= y < img_raw.shape[0]:
                        B, G, R = img_raw[y, x]
                        print(f"{name + ' Pixel RGB':<22}: (R : {R},  G : {G},  B : {B})")

                    else:
                        # 이미지 범위를 벗어난 경우, 해당 메시지를 출력
                        print(f"{name + ' Pixel RGB':<22}: Out of image bounds")

                print("====================================", end="\n\n")

                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # 이미지에 랜드마크 마킹
                for i in range(5):
                    x = int(b[5 + i * 2])
                    y = int(b[6 + i * 2])

                    if 0 <= x < img_raw.shape[1] and 0 <= y < img_raw.shape[0]:
                        color = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)][i]
                        cv2.circle(img_raw, (x, y), 1, color, 4)

                # 픽셀 포즈 정보 출력
                print("Pixel Pose Information")
                print("====================================")

                for i, name in enumerate(landmark_names):
                    x = int(b[5 + i * 2])
                    y = int(b[6 + i * 2])

                    # 좌표가 이미지 범위 내에 있는지 확인 후 출력
                    if 0 <= x < img_raw.shape[1] and 0 <= y < img_raw.shape[0]:
                        print(f"{name + ' Pixel pose':<22}: (x : {x},  y : {y})")

                    else:
                        # 이미지 범위를 벗어난 경우, 해당 메시지를 출력
                        print(f"{name + ' Pixel pose':<22}: Out of image bounds")

                print("====================================", end="\n\n")

                # Pose Relationship 출력
                print("Pose Relationship")
                print("====================================")

                # 랜드마크 좌표와 이미지 범위 내 여부 확인
                landmarks = [(int(b[5 + i * 2]), int(b[6 + i * 2])) for i in range(5)]
                in_image_bounds = [0 <= landmarks[i][0] < img_raw.shape[1] and 0 <= landmarks[i][1] < img_raw.shape[0]
                                   for i in range(5)]

                # 거리 관계 정의
                relationships = [
                    ("Distance Between Eyes", 0, 1),
                    ("Distance Between Right Eye and Nose", 0, 2),
                    ("Distance Between Left Eye and Nose", 1, 2),
                    ("Distance Between Right Mouth and Nose", 3, 2),
                    ("Distance Between Left Mouth and Nose", 4, 2),
                    ("Distance Between Mouths", 3, 4),
                ]

                # 거리 계산 및 출력
                for name, idx1, idx2 in relationships:
                    if in_image_bounds[idx1] and in_image_bounds[idx2]:
                        x1, y1 = landmarks[idx1]
                        x2, y2 = landmarks[idx2]
                        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        print(f"{name:<40}: {distance:.4f}")
                    else:
                        print(f"{name:<40}: Out of image bounds")

                print("====================================", end="\n\n")
                ## test

            cv2.imshow("Result", img_raw)
            print("TIME : ", time.time() - start, end="\n\n")
            cv2.waitKey(1)


if __name__ == "__main__":
    try:
        fddb = ros_fddb()

    except rospy.ROSInterruptException:
        pass