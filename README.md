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
I use ROS Noetic for real-time inference through a RGB camera.  
```shell
python3 ros_onny.py
```

<br/>

## Result
|                |     Base model    |   onnx model  | onnx model(optimization) |
|----------------|-------------------|---------------|--------------------------|
| Inference Time | 3,700 ~ 4,200(ms) | 520 ~ 570(ms) |       490 ~ 530(ms)      |

