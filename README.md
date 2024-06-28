# RetinaFace Optimization with Onnx & Quantization
The purpose of this reasearch is to investigate whether the RetinaFace model, utilizing MobileNet0.25 as its backbone, can operate in real-time on a Raspberry Pi 4 environment.
I have installed Ubuntu 20.04 server on a Raspberry Pi 4 with 4GB RAM and then set up the Ubuntu Desktop environment. This setup is running on an armv7l architecture(32 bit).

<br/>

## Table of Contents
- [Raspberry Pi 4 Environment Setting](#raspberry-pi-4-environment-setting)
  - [Pytorch Setting](#pytorch-setting-32bit-armv7l-standard)
  - [Onnxruntime Setting](#onnxruntime-setting)
- [Download RetinaFace_pytorch](#download-retinaface-pytorch)
  - [Workspace Setting](#workspace-setting)
- [Convert To Onnx](#convert-to-onnx)
- [Evaluation](#evaluation)
- [Result](#result)

<br/>

## Raspberry Pi 4 Environment Setting
### Pytorch Setting (32bit armv7l standard)
1. Check Raspberry OS
```shell
uname -a

# result
-->
```

<br/>

2. Check python version
If u install Ubuntu 20.04, your default python version is 3.8.x 
```shell
python3 --version

# result
--> python3.8.12
```

<br/>

3. Build Dependencies
```shell
sudo apt install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools
```

<br/>

4. Build Options
```shell
export NO_CUDA=1
export NO_DISTRIBUTED=1
export NO_MKLDNN=1 
export BUILD_TEST=0
export MAX_JOBS=<num>
```
- Set MAX_JOBS to an appropriate number based on your Raspberry Pi's RAM.
- In my case, I used a Raspberry Pi 4 with 4GB RAM and initially set **MAX_JOBS=4.**
- However, I encountered issues where the build process would hang.
- Therefore, I set **MAX_JOBS=2**, which resolved the issue.

<br/>

5. Install pytorch & torchvision  
On a Raspberry Pi, installing PyTorch and torchvision using the pip install command is limited. Therefore, you need to manually install them using .whl files.
```shell
# pytorch
git clone https://github.com/pytorch/pytorch --recursive && cd pytorch
git checkout v1.7.0
git submodule update --init --recursive
python setup.py bdist_wheel

# torchvision
git clone https://github.com/pytorch/vision && cd vision
git checkout v0.8.1
git submodule update --init --recursive
python setup.py bdist_wheel
```

<br/>

6. check
```shell
python3

import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
```

<br/>

### Onnxruntime Setting
1. Check Raspberry Pi os
```shell
cat /etc/debian_version

# result
--> bullseye
```

<br/>

2. Download whl file  
- Following the https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux
- Install the whl file compatible with the Python version and OS environment
  
ex)
- Environment
  - OS : bullseye
  - python : 3.8.12
  - 32bit(armv7l)
 
- Download **wheels/bullseye/onnxruntime-1.9.1-cp38-cp38-linux_armv7l.whl**
- cp38 = python3.8 / armv7l = 32bit

<br/>

3. Install whl file
```shell
pip3 install onnxruntime-1.9.1-cp38-cp38-linux_armv7l.whl
```

<br/>

4. Check Install
```shell
import onnxruntime as ort
print(ort.__version__)
```

<br/>

## Download RetinaFace pytorch
### Workspace Setting  
1. Use git clone to download RetinaFace
```shell
mkdir -p <your_workspace>/src
cd <your_workspace>/src
git clone https://github.com/biubug6/Pytorch_Retinaface
```

<br/>

2. Install dependencies
```shell
pip3 install pyyaml
pip3 install matplotlib
pip3 install numpy
pip3 install opencv-python
```

<br/>

### Modify Code
Because 

1. modify prior_box.py
```shell
# Before
class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"


# After
class PriorBox(object):
    def __init__(self, cfg, format:str="tensor", image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"
        self.__format = format

```
```shell
# Before
    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


# After
    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        if self.__format == "tensor":
            output = torch.Tensor(anchors).view(-1, 4)
        elif self.__format == "numpy":
            output = np.array(anchors).reshape(-1, 4)
        else:
            print(TypeError(("ERROR: INVALID TYPE OF FORMAT")))

        if self.clip:
            if self.__format == "tensor":
                output.clamp_(max=1, min=0)
            else:
                output = np.clip(output, 0, 1)

        return output
```

<br/>

2. Modify box_utils.py
```shell
# Before
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


# After
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = None
    if isinstance(loc, torch.Tensor) and isinstance(priors, torch.Tensor):
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)

    elif isinstance(loc, np.ndarray) and isinstance(priors, np.ndarray):
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)

    else:
        print(type(loc), type(priors))
        print(TypeError("ERROR: INVALID TYPE OF BOUNDING BOX"))

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
```
```shell
# Before
def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


# After
def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = None
    if isinstance(pre, torch.Tensor) and isinstance(priors, torch.Tensor):
        landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), dim=1)

    elif isinstance(pre, np.ndarray) and isinstance(priors, np.ndarray):
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                                 ), axis=1)

    else:
        print(TypeError("ERROR: INVALID TYPE OF LANDMARKS"))

    return landms
```

<br/>

## Convert To Onnx
```shell
from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--long_side', default=640,
                    help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = cfg_mnet if args.network == "mobile0.25" else cfg_re50

    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)

    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # ------------------------ export -----------------------------
    output_onnx = './weights/FaceDetector.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input"]
    output_names = ["bbox", "confidence", "landmark"]
    inputs = torch.randn(1, 3, 480, 640).to(device)

    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "bbox": {1: "num_boxes"},
        "confidence": {1: "num_boxes"},
        "landmark": {1: "num_boxes"}
    }

    torch.onnx.export(net, inputs, output_onnx, export_params=True, verbose=False,
                      input_names=input_names, output_names=output_names,
                      opset_version=11, dynamic_axes=dynamic_axes)

    # ------------------------ optimize -----------------------------
    print("==> Optimizing the exported ONNX model")
    optimized_model_path = './weights/optimized_FaceDetector.onnx'

    # Load the exported ONNX model
    onnx_model = onnx.load(output_onnx)

    # Optimize the model using ONNX Runtime
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(output_onnx, sess_options)

    # Save the optimized model
    onnx.save(onnx_model, optimized_model_path)
    print(f"Optimized ONNX model saved at '{optimized_model_path}'")
```

<br/>
## Evaluation
```shell
import csv
import cv2
import time
import math
import torch
import rospy
import argparse
import onnxruntime
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
parser.add_argument('--onnx_model', default='./weights/mobilenet0.25.onnx',
                    type=str, help='ONNX model file path to open')
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
```

<br/>

## Result

# My Project

## Table Example
| Name     | Age | City       |
|----------|-----|------------|
| John Doe | 25  | New York   |
| Jane Doe | 28  | Los Angeles|
| Sam Smith| 22  | Chicago    |


