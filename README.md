# Pytorch-Memory-Utils

These codes can help you to detect your GPU memory during training with Pytorch.
- Calculate the memory usage of a single model
```
Model Sequential : params: 0.450304M
Model Sequential : intermedite variables: 336.089600 M (without backward)
Model Sequential : intermedite variables: 672.179200 M (with backward)
```
- Track the amount of GPU memory
```
# 08-Jun-18-17:56:51-gpu_mem_prof

At __main__ <module>: line 39                        Total Used Memory:399.4  Mb
At __main__ <module>: line 40                        Total Used Memory:992.5  Mb
+ __main__ <module>: line 40                         (1, 1, 682, 700)     1.82 M <class 'torch.Tensor'>
+ __main__ <module>: line 40                         (1, 3, 682, 700)     5.46 M <class 'torch.Tensor'>
At __main__ <module>: line 126                       Total Used Memory:1088.5 Mb
+ __main__ <module>: line 126                        (64, 64, 3, 3)       0.14 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (128, 64, 3, 3)      0.28 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (128, 128, 3, 3)     0.56 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (64, 3, 3, 3)        0.00 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (256, 256, 3, 3)     2.25 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (512, 256, 3, 3)     4.5 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (512, 512, 3, 3)     9.0 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (64,)                0.00 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (1, 3, 682, 700)     5.46 M <class 'torch.Tensor'>
+ __main__ <module>: line 126                        (128,)               0.00 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (256,)               0.00 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (512,)               0.00 M <class 'torch.nn.parameter.Parameter'>
+ __main__ <module>: line 126                        (3,)                 1.14 M <class 'torch.Tensor'>
+ __main__ <module>: line 126                        (256, 128, 3, 3)     1.12 M <class 'torch.nn.parameter.Parameter'>
```
