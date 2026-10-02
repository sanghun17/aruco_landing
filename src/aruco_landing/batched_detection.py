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
        if backend == 'gpu-experimental':
            from aruco_landing.gpu_aruco import GpuArucoDetector
            self.gpu = GpuArucoDetector(self.detectors[0].dictionary)
        elif backend != 'cpu':
            raise ValueError('unknown detector backend: '+backend)
        cv2.setNumThreads(1)

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
            corners_ids = []
            for detector, image in zip(self.detectors, images):
                corners, ids, _ = detector.detector.detectMarkers(image)
                corners_ids.append((corners, [] if ids is None else ids.flatten().tolist()))
        detection_end = time.perf_counter()
        results = []
        for detector, (corners, ids) in zip(self.detectors, corners_ids):
            try:
                results.append(detector.estimate(corners, ids, self.K, self.D))
            except cv2.error:
                results.append(None)
        end = time.perf_counter()
        return results, dict(device_and_transfer_s=transfer_end-start,
                             detect_s=detection_end-transfer_end, pnp_s=end-detection_end,
                             transferred_bytes=transferred,
                             markers=sum(len(ids) for _,ids in corners_ids))
