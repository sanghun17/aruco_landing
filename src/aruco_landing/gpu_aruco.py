"""Experimental full-device 4x4/6x6 ArUco detector for the controlled Isaac scene.

Thresholding, connected components, quad/edge fitting and dictionary decoding
run in CUDA. Only counts, IDs and four corners cross to CPU. This is NOT an
OpenCV-equivalent detector: it assumes bright white paper/dark ink and uses a
fixed threshold. Keep the reference backend for evaluation until image/pose
accuracy and end-to-end throughput have been validated on rendered frames.
"""
import ctypes
import os
from pathlib import Path
import cv2
import numpy as np


def build_library(output, nvcc=None):
    import shutil
    import subprocess
    compiler = nvcc or shutil.which('nvcc')
    if not compiler:
        raise RuntimeError('nvcc missing; compile on a CUDA toolkit host and set ARUCO_CUDA_LIBRARY')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    source = Path(__file__).parent/'cuda/aruco.cu'
    subprocess.run([compiler,'-O3','-std=c++14','--shared','-Xcompiler','-fPIC',
        '--cudart','static','-gencode','arch=compute_75,code=sm_75',
        '-gencode','arch=compute_75,code=compute_75',str(source),'-o',str(output)],check=True)
    return output


class GpuArucoDetector:
    def __init__(self, dictionary, library=None, capacity=256):
        import torch
        shape = (dictionary.markerSize, len(dictionary.bytesList))
        if shape not in ((4,100),(6,50),(6,587)):
            raise ValueError('experimental CUDA backend supports DICT_4X4_100, DICT_6X6_50 and DICT_APRILTAG_36h11')
        path = library or os.environ.get('ARUCO_CUDA_LIBRARY')
        if not path or not Path(path).is_file():
            raise RuntimeError('set ARUCO_CUDA_LIBRARY to a compiled CUDA detector library')
        self.lib = ctypes.CDLL(str(path))
        self.lib.aruco_create.restype = ctypes.c_void_p
        self.lib.aruco_create.argtypes = [ctypes.c_int]*4
        self.lib.aruco_destroy.argtypes = [ctypes.c_void_p]
        self.dynamic_code_count = shape == (6,587)
        self.detect_function = (self.lib.aruco_detect6_dictionary if self.dynamic_code_count
                                else self.lib.aruco_detect if shape[0] == 4 else self.lib.aruco_detect6)
        self.detect_function.argtypes = [ctypes.c_void_p]*5 + ([ctypes.c_int] if self.dynamic_code_count else []) + [ctypes.c_int,ctypes.c_void_p]
        self.detect_function.restype = ctypes.c_int
        codes = []
        for marker_id in range(len(dictionary.bytesList)):
            bits = cv2.aruco.generateImageMarker(dictionary,marker_id,shape[0]+2)[1:-1,1:-1] > 0
            for rotation in range(4):
                code = sum(int(b)<<i for i,b in enumerate(np.rot90(bits,rotation).flat))
                codes.append(code)
        self.codes = torch.tensor(codes,dtype=torch.int32 if shape[0] == 4 else torch.int64,device='cuda')
        self.capacity = capacity
        self.handle = None
        self.shape = None
        self.device = self.codes.device

    def close(self):
        if self.handle:
            self.lib.aruco_destroy(self.handle)
            self.handle = None

    def detect(self, rgb):
        import torch
        if rgb.device != self.device or rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[-1] not in (3,4):
            raise ValueError('expected CUDA uint8 [N,H,W,3 or 4] on detector device')
        rgb = rgb.contiguous()
        n,h,w,c = rgb.shape
        if self.shape != (n,h,w):
            self.close()
            self.handle = self.lib.aruco_create(n,h,w,self.capacity)
            if not self.handle: raise RuntimeError('CUDA detector allocation failed')
            self.shape = (n,h,w)
        rows = torch.empty((n,self.capacity,9),dtype=torch.float32,device=rgb.device)
        counts = torch.empty(n,dtype=torch.int32,device=rgb.device)
        stream = torch.cuda.current_stream(rgb.device).cuda_stream
        arguments = [self.handle,rgb.data_ptr(),rows.data_ptr(),counts.data_ptr(),self.codes.data_ptr()]
        if self.dynamic_code_count:
            arguments.append(self.codes.numel())
        error = self.detect_function(*arguments,c,stream)
        if error: raise RuntimeError('CUDA detector launch failed: '+str(error))
        host_counts = counts.cpu().numpy()
        if np.any(host_counts < 0) or np.any(host_counts > self.capacity):
            raise RuntimeError('CUDA detector candidate/output overflow; reject frame')
        mask = torch.arange(self.capacity,device=rgb.device)[None,:] < counts[:,None]
        compact = rows[mask].cpu().numpy()
        batches, offset = [],0
        for count in host_counts:
            part = compact[offset:offset+count]
            batches.append(([r[1:].reshape(1,4,2) for r in part],part[:,0].astype(int).tolist()))
            offset += count
        return batches, host_counts.nbytes+compact.nbytes
