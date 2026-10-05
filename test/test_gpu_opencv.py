"""Reference-oracle checks; run explicitly on one healthy GPU with a built library."""
import os
import cv2
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(not os.environ.get('ARUCO_OPENCV_CUDA_LIBRARY'),
                            reason='requires explicitly built CUDA compatibility library')


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
