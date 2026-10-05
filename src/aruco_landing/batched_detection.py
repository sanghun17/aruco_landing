"""Camera GPU tensors to reference ArUco poses, with explicit transfer accounting."""
import time
import cv2
import numpy as np
from aruco_landing.physical_pad import PhysicalPadDetector


class BatchedPadDetector:
    def __init__(self, manifest, num_envs, K, backend='cpu'):
        self.detectors = [PhysicalPadDetector(manifest) for _ in range(num_envs)]
        self.K, self.D = np.asarray(K, np.float64), np.zeros(5)
        self.backend = backend
        self.gpu = None
        self.nested = None
        self.executor = None
        # Includes decoded markers even when pose estimation subsequently fails.
        self.last_detected_ids = [[] for _ in range(num_envs)]
        if backend == 'gpu-experimental':
            from aruco_landing.gpu_aruco import GpuArucoDetector
            self.gpu = GpuArucoDetector(self.detectors[0].dictionary)
        elif backend == 'gpu-opencv-compat':
            from aruco_landing.gpu_opencv import GpuOpenCVDetector
            self.gpu = GpuOpenCVDetector(self.detectors[0].dictionary, self.detectors[0].params)
        elif backend == 'cpu-nested-apriltag':
            from concurrent.futures import ThreadPoolExecutor
            from aruco_landing.nested_apriltag import NestedAprilTagTracker
            if manifest['dictionary'] != 'DICT_APRILTAG_36h11':
                raise ValueError('nested tracker requires the original AprilTag 36h11 dictionary')
            parents = [m for m in manifest['markers'] if m.get('render_cutouts_ids')]
            if len(parents)!=1:
                raise ValueError('nested tracker requires one parent and declared children')
            self.nested = [NestedAprilTagTracker(detector,parents[0]['render_cutouts_ids']) for detector in self.detectors]
            self.executor = ThreadPoolExecutor(max_workers=min(8,num_envs))
        elif backend == 'cpu':
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=min(8,num_envs))
        else:
            raise ValueError('unknown detector backend: '+backend)
        cv2.setNumThreads(1)

    def reset(self):
        if self.nested:
            for tracker in self.nested: tracker.reset()

    def _nested_detect(self, pair):
        tracker,image = pair
        return tracker.detect(image,self.K,self.D)

    @staticmethod
    def _cpu_detect(pair):
        detector, image = pair
        corners, ids, _ = detector.detector.detectMarkers(image)
        return corners, [] if ids is None else ids.flatten().tolist()

    def detect(self, rgb):
        import torch
        start = time.perf_counter()
        if self.gpu:
            corners_ids, transferred = self.gpu.detect(rgb)
            transfer_end = time.perf_counter()
        else:
            # Isaac camera is RGB(A). Compute one grayscale channel on-device;
            # transfer a single batch, rather than N ROS messages/RPC images.
            gray = ((rgb[...,:3].to(torch.int32) *
                     torch.tensor([77,150,29], device=rgb.device,dtype=torch.int32)).sum(-1,dtype=torch.int32) >> 8).to(torch.uint8)
            images = gray.cpu().numpy()
            transferred = images.nbytes
            transfer_end = time.perf_counter()
            if self.nested:
                corners_ids = list(self.executor.map(self._nested_detect,zip(self.nested,images)))
            else:
                # Each environment owns an independent OpenCV detector. Ordered
                # map keeps observations aligned with their simulation slots.
                corners_ids = list(self.executor.map(self._cpu_detect,zip(self.detectors,images)))
        detection_end = time.perf_counter()
        self.last_detected_ids = [list(ids) for _, ids in corners_ids]
        results = []
        for detector, (corners, ids) in zip(self.detectors, corners_ids):
            try:
                results.append(detector.estimate(corners, ids, self.K, self.D))
            except cv2.error:
                results.append(None)
        if self.nested:
            for tracker,observation in zip(self.nested,results): tracker.update(observation)
        end = time.perf_counter()
        return results, dict(device_and_transfer_s=transfer_end-start,
                             detect_s=detection_end-transfer_end, pnp_s=end-detection_end,
                             transferred_bytes=transferred,
                             markers=sum(len(ids) for _,ids in corners_ids))
