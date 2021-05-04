# Pytorch-Memory-Utils

These codes can help you to detect your GPU memory during training with Pytorch.

A blog about this tool and explain the details : https://oldpan.me/archives/pytorch-gpu-memory-usage-track

# Usage:

Put ``modelsize_estimate.py`` or ``gpu_mem_track.py`` under your current working directory and import them.

## The following is the print content.

- Calculate the memory usage of a single model
```
Model Sequential : params: 0.450304M
Model Sequential : intermedite variables: 336.089600 M (without backward)
Model Sequential : intermedite variables: 672.179200 M (with backward)
```
- Track the amount of GPU memory usage
```markdown
# 30-Apr-21-20:25:29-gpu_mem_track.txt

GPU Memory Track | 30-Apr-21-20:25:29 | Total Tensor Used Memory:0.0    Mb Total Used Memory:0.0    Mb


At main.py line 10: <module>                          Total Tensor Used Memory:0.0    Mb Total Allocated Memory:0.0    Mb

+ | 1 * Size:(64, 64, 3, 3)       | Memory: 0.1406 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 1 * Size:(128, 128, 3, 3)     | Memory: 0.5625 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 1 * Size:(256, 128, 3, 3)     | Memory: 1.125 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 1 * Size:(512, 256, 3, 3)     | Memory: 4.5 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 3 * Size:(256, 256, 3, 3)     | Memory: 6.75 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 8 * Size:(512,)               | Memory: 0.0156 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 2 * Size:(64,)                | Memory: 0.0004 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 7 * Size:(512, 512, 3, 3)     | Memory: 63.0 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 4 * Size:(256,)               | Memory: 0.0039 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 1 * Size:(128, 64, 3, 3)      | Memory: 0.2812 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 2 * Size:(128,)               | Memory: 0.0009 M | <class 'torch.nn.parameter.Parameter'> | torch.float32
+ | 1 * Size:(64, 3, 3, 3)        | Memory: 0.0065 M | <class 'torch.nn.parameter.Parameter'> | torch.float32

At main.py line 12: <module>                          Total Tensor Used Memory:76.4   Mb Total Allocated Memory:76.4   Mb

+ | 1 * Size:(60, 3, 512, 512)    | Memory: 180.0 M | <class 'torch.Tensor'> | torch.float32
+ | 1 * Size:(40, 3, 512, 512)    | Memory: 120.0 M | <class 'torch.Tensor'> | torch.float32
+ | 1 * Size:(30, 3, 512, 512)    | Memory: 90.0 M | <class 'torch.Tensor'> | torch.float32

At main.py line 18: <module>                          Total Tensor Used Memory:466.4  Mb Total Allocated Memory:466.4  Mb

+ | 1 * Size:(120, 3, 512, 512)   | Memory: 360.0 M | <class 'torch.Tensor'> | torch.float32
+ | 1 * Size:(80, 3, 512, 512)    | Memory: 240.0 M | <class 'torch.Tensor'> | torch.float32

At main.py line 23: <module>                          Total Tensor Used Memory:1066.4 Mb Total Allocated Memory:1066.4 Mb

- | 1 * Size:(40, 3, 512, 512)    | Memory: 120.0 M | <class 'torch.Tensor'> | torch.float32
- | 1 * Size:(120, 3, 512, 512)   | Memory: 360.0 M | <class 'torch.Tensor'> | torch.float32

At main.py line 29: <module>                          Total Tensor Used Memory:586.4  Mb Total Allocated Memory:586.4  Mb
```

## How to use

### Track the amount of GPU memory usage
simple example:

```python
import torch

from torchvision import models
from gpu_mem_track import MemTracker

device = torch.device('cuda:0')

gpu_tracker = MemTracker()         # define a GPU tracker

gpu_tracker.track()                     # run function between the code line where uses GPU
cnn = models.vgg19(pretrained=True).features.to(device).eval()
gpu_tracker.track()                     # run function between the code line where uses GPU

dummy_tensor_1 = torch.randn(30, 3, 512, 512).float().to(device)  # 30*3*512*512*4/1024/1024 = 90.00M
dummy_tensor_2 = torch.randn(40, 3, 512, 512).float().to(device)  # 40*3*512*512*4/1024/1024 = 120.00M
dummy_tensor_3 = torch.randn(60, 3, 512, 512).float().to(device)  # 60*3*512*512*4/1024/1024 = 180.00M

gpu_tracker.track()

dummy_tensor_4 = torch.randn(120, 3, 512, 512).float().to(device)  # 120*3*512*512*4/1024/1024 = 360.00M
dummy_tensor_5 = torch.randn(80, 3, 512, 512).float().to(device)  # 80*3*512*512*4/1024/1024 = 240.00M

gpu_tracker.track()

dummy_tensor_4 = dummy_tensor_4.cpu()
dummy_tensor_2 = dummy_tensor_2.cpu()
gpu_tracker.clear_cache() # or torch.cuda.empty_cache()

gpu_tracker.track()
```
This will output a ``.txt`` to current dir and the content of output is above(print content).

# FAQs

1. Why Total Tensor Used Memory is much smaller than Total Allocated Memory?

* Total Allocated Memory is the peak of the memory usage. When you delete some tensors, PyTorch will not release the space to the device, until you call ``gpu_tracker.clear_cache()`` like the example script.

* The cuda kernel will take some space. See https://github.com/pytorch/pytorch/issues/12873

2. Why does Total Allocated Memory stay unchanged?

* See Q1.

3. I deleted some tensors. Why are they not deleted in tracker's output?

* Make sure that you have released all the references to the tensor object. Then you can call "import gc; gc.collect()" and tell python to collect the unreferenced tensor.

# REFERENCE
Part of the code is referenced from:

http://jacobkimmel.github.io/pytorch_estimating_model_size/ 
https://gist.github.com/MInner/8968b3b120c95d3f50b8a22a74bf66bc

