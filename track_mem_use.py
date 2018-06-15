import datetime
import linecache
import os

import gc
import pynvml
import torch
import numpy as np


print_tensor_sizes = True
last_tensor_sizes = set()
gpu_profile_fn = f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_prof.txt'

# if 'GPU_DEBUG' in os.environ:
#     print('profiling gpu usage to ', gpu_profile_fn)

lineno = None
func_name = None
filename = None
module_name = None

# fram = inspect.currentframe()
# func_name = fram.f_code.co_name
# filename = fram.f_globals["__file__"]
# ss = os.path.dirname(os.path.abspath(filename))
# module_name = fram.f_globals["__name__"]


def gpu_profile(frame, event):
    # it is _about to_ execute (!)
    global last_tensor_sizes
    global lineno, func_name, filename, module_name

    if event == 'line':
        try:
            # about _previous_ line (!)
            if lineno is not None:
                pynvml.nvmlInit()
                # handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ['GPU_DEBUG']))
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                line = linecache.getline(filename, lineno)
                where_str = module_name+' '+func_name+':'+' line '+str(lineno)

                with open(gpu_profile_fn, 'a+') as f:
                    f.write(f"At {where_str:<50}"
                            f"Total Used Memory:{meminfo.used/1024**2:<7.1f}Mb\n")

                    if print_tensor_sizes is True:
                        for tensor in get_tensors():
                            if not hasattr(tensor, 'dbg_alloc_where'):
                                tensor.dbg_alloc_where = where_str
                        new_tensor_sizes = {(type(x), tuple(x.size()), np.prod(np.array(x.size()))*4/1024**2,
                                             x.dbg_alloc_where) for x in get_tensors()}
                        for t, s, m, loc in new_tensor_sizes - last_tensor_sizes:
                            f.write(f'+ {loc:<50} {str(s):<20} {str(m)[:4]} M {str(t):<10}\n')
                        for t, s, m, loc in last_tensor_sizes - new_tensor_sizes:
                            f.write(f'- {loc:<50} {str(s):<20} {str(m)[:4]} M {str(t):<10}\n')
                        last_tensor_sizes = new_tensor_sizes
                pynvml.nvmlShutdown()

            # save details about line _to be_ executed
            lineno = None

            func_name = frame.f_code.co_name
            filename = frame.f_globals["__file__"]
            if (filename.endswith(".pyc") or
                    filename.endswith(".pyo")):
                filename = filename[:-1]
            module_name = frame.f_globals["__name__"]
            lineno = frame.f_lineno

            return gpu_profile

        except Exception as e:
            print('A exception occured: {}'.format(e))

    return gpu_profile


def get_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            else:
                continue
            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            print('A exception occured: {}'.format(e))
            
            
            
