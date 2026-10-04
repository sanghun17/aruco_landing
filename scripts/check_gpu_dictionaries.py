#!/usr/bin/env python3
"""Small CUDA decoding and 4x4 binary compatibility check; no Isaac rendering."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from aruco_landing.gpu_aruco import GpuArucoDetector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',required=True)
    parser.add_argument('--previous-library',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=dict(scope='synthetic decoding only; no renderer or landing accuracy claim',cases=[])
    for name,ids in [('DICT_4X4_100',[0,17,43,99]),('DICT_6X6_50',[0,1,2,3,4,19,43,49])]:
        dictionary=cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,name))
        detector=GpuArucoDetector(dictionary,args.library)
        previous=GpuArucoDetector(dictionary,args.previous_library) if dictionary.markerSize==4 else None
        cpu=cv2.aruco.ArucoDetector(dictionary)
        for mid in ids:
            for rotation in range(4):
                image=np.full((160,160),255,np.uint8)
                image[40:120,40:120]=np.rot90(cv2.aruco.generateImageMarker(dictionary,mid,80),rotation)
                rgb=torch.from_numpy(np.repeat(image[None,...,None],3,axis=-1)).cuda()
                found,transferred=detector.detect(rgb)
                assert found[0][1]==[mid],(name,mid,rotation,found[0][1])
                corners,ref_ids,_=cpu.detectMarkers(image)
                assert ref_ids.flatten().tolist()==[mid]
                corner_delta=float(np.linalg.norm(found[0][0][0]-corners[0],axis=2).max())
                assert corner_delta<=1.,(name,mid,rotation,corner_delta)
                if previous:
                    old,_=previous.detect(rgb)
                    assert old[0][1]==found[0][1]
                    np.testing.assert_array_equal(old[0][0][0],found[0][0][0])
                report['cases'].append(dict(dictionary=name,id=mid,rotation=rotation,
                    reference_corner_max_delta_px=corner_delta,transferred_bytes=transferred,
                    previous_4x4_exact_match=previous is not None))
        if previous: previous.close()
        detector.close()
    report['passed']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Passed',len(report['cases']),'dictionary/ID/rotation cases; 4x4 corners exactly match previous binary')


if __name__=='__main__': main()
