"""OpenCV-compatible CUDA image processing with compact CPU verification.

The image stays on CUDA: replicated-border adaptive thresholds, RETR_LIST /
CHAIN_APPROX_NONE contours, polygon approximation and candidate patches are
computed on-device. Stable grouping, dictionary identification and the actual
OpenCV cornerSubPix remain on CPU to preserve their reference behavior.
This is an opt-in compatibility prototype, not an all-CUDA detector.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np


def build_library(output, nvcc=None):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler = nvcc or shutil.which('nvcc')
    if not compiler:
        raise RuntimeError('CUDA toolkit nvcc is required')
    subprocess.run([compiler, '-O3', '--fmad=false', '-std=c++14', '--shared',
                    '-Xcompiler', '-fPIC', '--cudart', 'static',
                    '-gencode', 'arch=compute_75,code=sm_75',
                    '-gencode', 'arch=compute_75,code=compute_75',
                    str(Path(__file__).parent/'cuda/opencv_candidates.cu'),
                    '-o', str(output)], check=True)
    return output


class GpuOpenCVDetector:
    def __init__(self, dictionary, params, library=None, chunk_size=16, capacity=4096):
        import torch
        path = library or os.environ.get('ARUCO_OPENCV_CUDA_LIBRARY')
        if not path or not Path(path).is_file():
            raise RuntimeError('set ARUCO_OPENCV_CUDA_LIBRARY to the compatibility library')
        if params.useAruco3Detection or params.cornerRefinementMethod not in (
                cv2.aruco.CORNER_REFINE_NONE, cv2.aruco.CORNER_REFINE_SUBPIX):
            raise ValueError('prototype supports classic ArUco with NONE/SUBPIX refinement')
        if not (1 <= params.cornerRefinementWinSize <= 10):
            raise ValueError('prototype supports corner windows 1..10')
        if chunk_size < 1 or capacity < 1:
            raise ValueError('chunk size and candidate capacity must be positive')
        self.dictionary, self.params = dictionary, params
        self.chunk_size, self.capacity = chunk_size, capacity
        self.lib = ctypes.CDLL(str(path))
        ptr, integer, real = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
        signatures = {
            'ocv_threshold': [ptr, ptr]+[integer]*7+[real, ptr],
            'ocv_candidates': [ptr]+[integer]*3+[ptr, integer, ptr, ptr, integer]+[real]*4+[ptr],
            'ocv_group': [ptr]+[integer]*5+[real, integer, real, integer, ptr, ptr],
            'ocv_warp': [ptr, integer, integer, ptr, ptr, ptr, integer, integer, ptr],
            'ocv_rois': [ptr, integer, integer, ptr, ptr, integer, integer, ptr],
        }
        for name, signature in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = signature, integer
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.last_stats = {}

    def _call(self, name, *args):
        import torch
        error = getattr(self.lib, name)(*args, torch.cuda.current_stream(self.device).cuda_stream)
        if error:
            raise RuntimeError(f'{name}: CUDA error {error}')

    def _candidates(self, gray):
        import torch
        import torch.nn.functional as F
        p = self.params
        n, h, w = gray.shape
        if (p.adaptiveThreshWinSizeMin < 3 or p.adaptiveThreshWinSizeStep < 1
                or p.adaptiveThreshWinSizeMax < p.adaptiveThreshWinSizeMin):
            raise ValueError('invalid adaptive threshold windows')
        levels = (p.adaptiveThreshWinSizeMax-p.adaptiveThreshWinSizeMin)//p.adaptiveThreshWinSizeStep+1
        largest = p.adaptiveThreshWinSizeMin+(levels-1)*p.adaptiveThreshWinSizeStep
        largest += not (largest & 1)
        pad = largest//2
        padded = F.pad(gray[:, None], (pad, pad, pad, pad), mode='replicate')[:, 0]
        # uint8 maximum sum must fit the signed int32 summed-area table.
        if (h+2*pad)*(w+2*pad)*255 >= 2**31:
            raise ValueError('image/window size exceeds integral table range')
        sat = F.pad(padded.cumsum(1, dtype=torch.int32).cumsum(2, dtype=torch.int32), (1,0,1,0)).contiguous()
        binary = torch.empty((n*levels, h+2, w+2), dtype=torch.int8, device=gray.device)
        self._call('ocv_threshold', sat.data_ptr(), binary.data_ptr(), n,h,w,pad,
                   p.adaptiveThreshWinSizeMin,p.adaptiveThreshWinSizeStep,levels,p.adaptiveThreshConstant)
        work_size = max(16, int(p.maxMarkerPerimeterRate*max(h,w))+8)
        work = torch.empty((n*levels, 3, work_size, 2), dtype=torch.int32, device=gray.device)
        quads = torch.empty((n*levels,self.capacity,4,2), dtype=torch.float32, device=gray.device)
        counts = torch.empty(n*levels, dtype=torch.int32, device=gray.device)
        self._call('ocv_candidates', binary.data_ptr(),h,w,n*levels,work.data_ptr(),work_size,
                   quads.data_ptr(),counts.data_ptr(),self.capacity,p.minMarkerPerimeterRate,
                   p.maxMarkerPerimeterRate,p.polygonalApproxAccuracyRate,p.minCornerDistanceRate)
        host_counts = counts.cpu().numpy()
        if np.any(host_counts < 0):
            raise RuntimeError('CUDA contour/candidate overflow; frame rejected rather than truncated')
        mask = torch.arange(self.capacity, device=gray.device)[None, :] < counts[:, None]
        compact = quads[mask].cpu().numpy()
        result, offset = [], 0
        for env in range(n):
            planes = []
            for level in range(levels):
                count = host_counts[env*levels+level]
                # RETR_LIST emits contours in reverse raster-discovery order.
                planes.append(compact[offset:offset+count][::-1])
                offset += count
            result.append(np.ascontiguousarray(np.concatenate(planes)))
        return result, host_counts.nbytes+compact.nbytes

    def _group(self, q, h, w):
        p = self.params
        metadata = np.empty((len(q),5),np.int32)
        close = np.empty(len(q),np.int32)
        count = self.lib.ocv_group(q.ctypes.data,len(q),h,w,self.dictionary.markerSize,p.markerBorderBits,
                                  p.minMarkerDistanceRate,p.minDistanceToBorder,p.minGroupDistance,
                                  int(p.detectInvertedMarker),metadata.ctypes.data,close.ctypes.data)
        return metadata[:count], close

    def _identify(self, patch):
        p = self.params
        cell, border = p.perspectiveRemovePixelPerCell, p.markerBorderBits
        cells = self.dictionary.markerSize+2*border
        cut = cell//2
        interior = patch[cut:patch.shape[0]-cut,cut:patch.shape[1]-cut] if cut else patch
        mean, std = cv2.meanStdDev(interior)
        if std[0,0] < p.minOtsuStdDev:
            ratios = np.full((cells,cells), int(mean[0,0]>127),np.float32)
        else:
            _, threshold = cv2.threshold(patch,125,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            margin = int(p.perspectiveRemoveIgnoredMarginPerCell*cell)
            if 2*margin >= cell:
                raise ValueError('cell margin removes all samples')
            squares = threshold.reshape(cells,cell,cells,cell).transpose(0,2,1,3)
            if margin:
                squares = squares[:,:,margin:cell-margin,margin:cell-margin]
            ratios = np.count_nonzero(squares,axis=(2,3)).astype(np.float32)
            ratios /= np.float32(squares.shape[2]*squares.shape[3])
        # 4.14 changed the reference detector to ratio-based decoding. In
        # particular 50/50 cells are ambiguous rather than binary black bits.
        # Use the runtime's own dictionary overload and border threshold;
        # converting ratios to binary here would silently preserve 4.13 rules.
        ratio_decoder = hasattr(p, 'validBitIdThreshold')
        threshold = np.float32(p.validBitIdThreshold) if ratio_decoder else np.float32(.5)
        if not ratio_decoder:
            ratios = (ratios > np.float32(.5)).astype(np.float32)
        border_mask = np.ones_like(ratios,bool)
        border_mask[border:-border,border:-border] = False
        errors = int(np.count_nonzero(ratios[border_mask] > threshold))
        inverted_errors = int(np.count_nonzero(ratios[border_mask] < np.float32(1)-threshold)) if ratio_decoder else int(np.count_nonzero(ratios[border_mask] <= threshold))
        if p.detectInvertedMarker and inverted_errors < errors:
            ratios = np.float32(1)-ratios
            errors = inverted_errors
        if errors > int(self.dictionary.markerSize**2*p.maxErroneousBitsInBorderRate):
            return None
        inner = np.ascontiguousarray(ratios[border:-border,border:-border])
        if ratio_decoder:
            okay, mid, rotation = self.dictionary.identify(inner,p.errorCorrectionRate,p.validBitIdThreshold)
        else:
            # Before 4.14 the detector assigned white only for a strict majority.
            bits = (inner > np.float32(.5)).astype(np.uint8)
            okay, mid, rotation = self.dictionary.identify(bits,p.errorCorrectionRate)
        return (mid,rotation) if okay else None

    def _chunk(self, gray):
        import torch
        p = self.params
        n,h,w = gray.shape
        quads, transferred = self._candidates(gray)
        trees = list(self.executor.map(lambda q:self._group(q,h,w),quads))
        all_q, envs, lookup = [], [], []
        for env,(q,(meta,close)) in enumerate(zip(quads,trees)):
            mappings = []
            for row in meta:
                indices = [row[0]]+close[row[3]:row[3]+row[4]].tolist()
                mapping = []
                for index in indices:
                    mapping.append(len(all_q));all_q.append(q[index]);envs.append(env)
                mappings.append(mapping)
            lookup.append(mappings)
        size = (self.dictionary.markerSize+2*p.markerBorderBits)*p.perspectiveRemovePixelPerCell
        patches = np.empty((0,size,size),np.uint8)
        if all_q:
            canonical = np.float32([[0,0],[size-1,0],[size-1,size-1],[0,size-1]])
            # CPU OpenCV supplies the exact reference homography and inverse.
            matrices = np.array([cv2.invert(cv2.getPerspectiveTransform(q,canonical))[1] for q in all_q])
            gm = torch.from_numpy(matrices).to(gray.device)
            ge = torch.tensor(envs,dtype=torch.int32,device=gray.device)
            gp = torch.empty((len(all_q),size,size),dtype=torch.uint8,device=gray.device)
            self._call('ocv_warp',gray.data_ptr(),h,w,gm.data_ptr(),ge.data_ptr(),gp.data_ptr(),size,len(all_q))
            patches = gp.cpu().numpy()
            transferred += patches.nbytes
        decoded = [self._identify(patch) for patch in patches]
        results = []
        for env,(meta,_) in enumerate(trees):
            found = [None]*len(meta);chosen = [None]*len(meta);was = np.zeros(len(meta),bool)
            counter,depth = 0,0
            while counter < len(meta):
                if depth > max(meta[:,2],default=-1):
                    raise RuntimeError('candidate tree traversal did not finish')
                at_depth = np.flatnonzero(meta[:,2]==depth)
                for v in at_depth:
                    was[v] = True
                    for index in lookup[env][v]:
                        if decoded[index] is not None:
                            found[v]=decoded[index];chosen[v]=all_q[index];break
                for v in at_depth:
                    if found[v] is not None:
                        parent = meta[v,1]
                        while parent != -1:
                            if not was[parent]:was[parent]=True;counter+=1
                            parent=meta[parent,1]
                    counter+=1
                depth+=1
            ids, corners = [], []
            for value,q in zip(found,chosen):
                if value is not None:
                    mid,rotation=value
                    ids.append(mid);corners.append(np.roll(q,rotation,axis=0).copy())
            results.append((corners,ids))
        if p.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_SUBPIX:
            transferred += self._refine(gray,results)
        return [( [q.reshape(1,4,2) for q in corners], ids) for corners,ids in results],transferred

    def _refine(self, gray, results):
        import torch
        p = self.params
        n,h,w = gray.shape
        # Retain original global coordinates when calling cornerSubPix. A sparse
        # host raster is filled only around each corner, preserving FP32 rounding
        # that changes when corners are translated to small local ROI coordinates.
        radius = 3*p.cornerRefinementWinSize+4
        size = radius*2+1
        records = []
        for env,(corners,_) in enumerate(results):
            for q in corners:
                for point in q:
                    x,y = np.floor(point).astype(int)
                    records.append((env,x-radius,y-radius))
        if not records:return 0
        gr = torch.tensor(records,dtype=torch.int32,device=gray.device)
        gp = torch.empty((len(records),size,size),dtype=torch.uint8,device=gray.device)
        self._call('ocv_rois',gray.data_ptr(),h,w,gr.data_ptr(),gp.data_ptr(),size,len(records))
        patches = gp.cpu().numpy()
        buffers = [np.zeros((h,w),np.uint8) for _ in range(n)]
        for (env,x,y),patch in zip(records,patches):
            xa,ya,xb,yb=max(0,x),max(0,y),min(w,x+size),min(h,y+size)
            buffers[env][ya:yb,xa:xb]=patch[ya-y:yb-y,xa-x:xb-x]
        for env,(corners,_) in enumerate(results):
            for q in corners:
                edges = q-np.roll(q,-1,axis=0)
                # Match OpenCV's four sequential FP32 length additions.
                perimeter=np.float32(0)
                for e in edges:perimeter=np.float32(perimeter+np.sqrt(np.float32(e@e)))
                module=np.float32(perimeter/np.float32(4*(self.dictionary.markerSize+2*p.markerBorderBits)))
                win=min(p.cornerRefinementWinSize,max(1,int(np.rint(p.relativeCornerRefinmentWinSize*module))))
                cv2.cornerSubPix(buffers[env],q,(win,win),(-1,-1),
                                 (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,p.cornerRefinementMaxIterations,p.cornerRefinementMinAccuracy))
        return patches.nbytes

    def detect_gray(self, gray):
        import torch
        if (gray.device != self.device or gray.dtype != torch.uint8 or gray.ndim != 3
                or gray.shape[-1] < 1 or gray.shape[-2] < 1):
            raise ValueError('expected CUDA uint8 [N,H,W] on the configured device')
        gray=gray.contiguous()
        result,transferred=[],0
        for start in range(0,len(gray),self.chunk_size):
            rows,count=self._chunk(gray[start:start+self.chunk_size]);result.extend(rows);transferred+=count
        self.last_stats={'transferred_bytes':transferred,'grayscale_frame_bytes':gray.numel(),
                         'full_images_downloaded':False,'cpu_stages':['candidate grouping','patch Otsu / dictionary identify','cornerSubPix','PnP (caller)']}
        return result,transferred

    def detect(self, rgb):
        import torch
        if rgb.device != self.device or rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[-1] not in (3,4):
            raise ValueError('expected CUDA uint8 [N,H,W,3 or 4] on the configured device')
        # Same grayscale input as BatchedPadDetector's CPU reference path.
        gray=((rgb[...,:3].to(torch.int32)*torch.tensor([77,150,29],device=rgb.device,dtype=torch.int32)).sum(-1,dtype=torch.int32)>>8).to(torch.uint8)
        return self.detect_gray(gray)

    def close(self):
        self.executor.shutdown(wait=True)
