# RetinaFace Optimization with Onnx & Quantization
The purpose of this reasearch is to investigate whether the RetinaFace model, utilizing MobileNet0.25 as its backbone, can operate in real-time on a Raspberry Pi 4 environment.
I have installed Ubuntu 20.04 server on a Raspberry Pi 4 with 4GB RAM and then set up the Ubuntu Desktop environment. This setup is running on an armv7l architecture(32 bit).

<br/>

## Table of Contents
- [Raspberry Pi 4 Environment Setting](#raspberry-pi-4-environment-setting)
  - [Pytorch Setting](#update-and-upgrade)
  - [Install Essential Tools](#install-essential-tools)
  - [Enable SSH](#enable-ssh)
  - [Configure Swap Space](#configure-swap-space)
- [Installation](#installation)
  - [Download the .whl files](#download-the-whl-files)
  - [Install the .whl files](#install-the-whl-files)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
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






