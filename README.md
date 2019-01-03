# Pytorch-Memory-Utils

These codes can help you to detect your GPU memory during training with Pytorch.

A blog about this tool and explain the details : https://oldpan.me/archives/pytorch-gpu-memory-usage-track

# Requirement:

```
pynvml(pip install nvidia-ml-py3)
```

## The following is the print content.

- Calculate the memory usage of a single model
```
Model Sequential : params: 0.450304M
Model Sequential : intermedite variables: 336.089600 M (without backward)
Model Sequential : intermedite variables: 672.179200 M (with backward)
```
- Track the amount of GPU memory usage
```markdown
# 12-Sep-18-21:48:45-gpu_mem_track.txt

GPU Memory Track | 12-Sep-18-21:48:45 | Total Used Memory:696.5  Mb

At __main__ <module>: line 13                        Total Used Memory:696.5  Mb

+ | 7 * Size:(512, 512, 3, 3)     | Memory: 66.060 M | <class 'torch.nn.parameter.Parameter'>
+ | 1 * Size:(512, 256, 3, 3)     | Memory: 4.7185 M | <class 'torch.nn.parameter.Parameter'>
+ | 1 * Size:(64, 64, 3, 3)       | Memory: 0.1474 M | <class 'torch.nn.parameter.Parameter'>
+ | 1 * Size:(128, 64, 3, 3)      | Memory: 0.2949 M | <class 'torch.nn.parameter.Parameter'>
+ | 1 * Size:(128, 128, 3, 3)     | Memory: 0.5898 M | <class 'torch.nn.parameter.Parameter'>
+ | 8 * Size:(512,)               | Memory: 0.0163 M | <class 'torch.nn.parameter.Parameter'>
+ | 3 * Size:(256, 256, 3, 3)     | Memory: 7.0778 M | <class 'torch.nn.parameter.Parameter'>
+ | 1 * Size:(256, 128, 3, 3)     | Memory: 1.1796 M | <class 'torch.nn.parameter.Parameter'>
+ | 2 * Size:(64,)                | Memory: 0.0005 M | <class 'torch.nn.parameter.Parameter'>
+ | 4 * Size:(256,)               | Memory: 0.0040 M | <class 'torch.nn.parameter.Parameter'>
+ | 2 * Size:(128,)               | Memory: 0.0010 M | <class 'torch.nn.parameter.Parameter'>
+ | 1 * Size:(64, 3, 3, 3)        | Memory: 0.0069 M | <class 'torch.nn.parameter.Parameter'>

At __main__ <module>: line 15                        Total Used Memory:1142.0 Mb

+ | 1 * Size:(60, 3, 512, 512)    | Memory: 188.74 M | <class 'torch.Tensor'>
+ | 1 * Size:(30, 3, 512, 512)    | Memory: 94.371 M | <class 'torch.Tensor'>
+ | 1 * Size:(40, 3, 512, 512)    | Memory: 125.82 M | <class 'torch.Tensor'>

At __main__ <module>: line 21                        Total Used Memory:1550.9 Mb

+ | 1 * Size:(120, 3, 512, 512)   | Memory: 377.48 M | <class 'torch.Tensor'>
+ | 1 * Size:(80, 3, 512, 512)    | Memory: 251.65 M | <class 'torch.Tensor'>

At __main__ <module>: line 26                        Total Used Memory:2180.1 Mb

- | 1 * Size:(120, 3, 512, 512)   | Memory: 377.48 M | <class 'torch.Tensor'> 
- | 1 * Size:(40, 3, 512, 512)    | Memory: 125.82 M | <class 'torch.Tensor'> 

At __main__ <module>: line 32                        Total Used Memory:1676.8 Mb
```

## How to use

### Track the amount of GPU memory usage
simple example:

```python
import torch
import inspect

from torchvision import models
from gpu_mem_track import  MemTracker

device = torch.device('cuda:0')

frame = inspect.currentframe()          # define a frame to track
gpu_tracker = MemTracker(frame)         # define a GPU tracker

gpu_tracker.track()                     # run function between the code line where uses GPU
cnn = models.vgg19(pretrained=True).features.to(device).eval()
gpu_tracker.track()                     # run function between the code line where uses GPU

dummy_tensor_1 = torch.randn(30, 3, 512, 512).float().to(device)  # 30*3*512*512*4/1000/1000 = 94.37M
dummy_tensor_2 = torch.randn(40, 3, 512, 512).float().to(device)  # 40*3*512*512*4/1000/1000 = 125.82M
dummy_tensor_3 = torch.randn(60, 3, 512, 512).float().to(device)  # 60*3*512*512*4/1000/1000 = 188.74M

gpu_tracker.track()

dummy_tensor_4 = torch.randn(120, 3, 512, 512).float().to(device)  # 120*3*512*512*4/1000/1000 = 377.48M
dummy_tensor_5 = torch.randn(80, 3, 512, 512).float().to(device)  # 80*3*512*512*4/1000/1000 = 251.64M

gpu_tracker.track()

dummy_tensor_4 = dummy_tensor_4.cpu()
dummy_tensor_2 = dummy_tensor_2.cpu()
torch.cuda.empty_cache()

gpu_tracker.track()
```
This will output a .txt to current dir and the content of output is above(print content).

# REFERENCE
Part of the code is referenced from:

http://jacobkimmel.github.io/pytorch_estimating_model_size/ 
https://gist.github.com/MInner/8968b3b120c95d3f50b8a22a74bf66bc

