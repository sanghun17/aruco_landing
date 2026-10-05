"""Reference-oracle checks; run explicitly on one healthy GPU with a built library."""
import os
import cv2
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(not os.environ.get('ARUCO_OPENCV_CUDA_LIBRARY'),
                            reason='requires explicitly built CUDA compatibility library')


def test_adaptive_threshold_pixels_match_reference_at_all_configured_scales():
    import torch
    import torch.nn.functional as F
    from aruco_landing.gpu_opencv import GpuOpenCVDetector
    from aruco_landing.physical_pad import PhysicalPadDetector
    estimator=PhysicalPadDetector({'dictionary':'DICT_4X4_100','markers':[]})
    images=np.random.default_rng(7).integers(0,256,(2,80,90),dtype=np.uint8)
    gray=torch.from_numpy(images).cuda()
    # PhysicalPadDetector uses windows 3,7,...,99 (max configured as 101).
    pad=49
    padded=F.pad(gray[:,None],(pad,pad,pad,pad),mode='replicate')[:,0]
    sat=F.pad(padded.cumsum(1,dtype=torch.int32).cumsum(2,dtype=torch.int32),(1,0,1,0)).contiguous()
    binary=torch.empty((50,82,92),dtype=torch.int8,device=gray.device)
    detector=GpuOpenCVDetector(estimator.dictionary,estimator.params)
    try:
        detector._call('ocv_threshold',sat.data_ptr(),binary.data_ptr(),2,80,90,pad,3,4,25,7.)
        actual=binary.cpu().numpy()
        for env in range(2):
            for level in range(25):
                reference=cv2.adaptiveThreshold(images[env],255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV,3+4*level,7.)//255
                np.testing.assert_array_equal(reference,actual[env*25+level,1:-1,1:-1])
    finally:
        detector.close()


@pytest.mark.parametrize('family,mid', [('DICT_4X4_100',92),('DICT_6X6_50',43),
                                       ('DICT_APRILTAG_36h11',166)])
def test_reference_ids_corners_across_scale_and_light(family,mid):
    import torch
    from aruco_landing.gpu_opencv import GpuOpenCVDetector
    from aruco_landing.physical_pad import PhysicalPadDetector
    estimator=PhysicalPadDetector({'dictionary':family,'markers':[]})
    d=estimator.dictionary
    images=[]
    for side in (23,75,220):
        marker=cv2.aruco.generateImageMarker(d,mid,side)
        im=np.full((360,360),230,np.uint8)
        start=(360-side)//2;im[start:start+side,start:start+side]=marker
        images.append(im)
        images.append(np.clip(im*.48+np.linspace(20,90,360)[None,:],0,255).astype(np.uint8))
    images.append(np.full((360,360),255,np.uint8))
    detector=GpuOpenCVDetector(d,estimator.params)
    try:
        gpu,amount=detector.detect_gray(torch.from_numpy(np.stack(images)).cuda())
        matches=0
        for image,(corners,ids) in zip(images,gpu):
            q,found,_=estimator.detector.detectMarkers(image)
            reference=[] if found is None else found.flatten().tolist()
            assert ids==reference
            for a,b in zip(corners,q):
                np.testing.assert_allclose(a,b,atol=1e-3,rtol=0)
                matches+=1
        assert matches>=4
        assert amount<np.stack(images).nbytes
    finally:
        detector.close()
