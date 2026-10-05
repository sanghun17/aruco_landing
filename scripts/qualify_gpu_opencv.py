#!/usr/bin/env python3
"""Compare the CUDA compatibility frontend with the actual CPU OpenCV detector.

Images are held on CUDA for both frontend timings. CPU timings include the
grayscale batch download and use a fixed worker pool. Rendering is excluded.
Saved native images may be provided by a JSON corpus of path/dictionary pairs.
No existing evaluation traces or defaults are changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from aruco_landing.gpu_opencv import GpuOpenCVDetector, build_library


def parameters():
    p=cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMax=101;p.adaptiveThreshWinSizeStep=4
    p.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize=3;p.cornerRefinementMaxIterations=50;p.cornerRefinementMinAccuracy=.01
    return p


def synthetic():
    cases=[];rng=np.random.default_rng(1701)
    for family,ids in [('DICT_4X4_100',[0,17,57,92]),('DICT_6X6_50',[1,19,43,49]),
                       ('DICT_APRILTAG_36h11',[6,166,586,24])]:
        d=cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,family))
        for index in range(24):
            image=np.full((720,720),255,np.uint8)
            side=[18,35,80,170,390,590][index//4]
            q=np.float32([[0,0],[side,0],[side,side],[0,side]])+(720-side)/2
            q+=rng.uniform(-.04*side,.04*side,(4,2)).astype(np.float32)
            marker=np.rot90(cv2.aruco.generateImageMarker(d,ids[index%4],160),index%4).copy()
            H=cv2.getPerspectiveTransform(np.float32([[0,0],[159,0],[159,159],[0,159]]),q)
            image=cv2.warpPerspective(marker,H,(720,720),borderValue=255)
            if index%3==0:image=cv2.GaussianBlur(image,(3,3),.7)
            if index%3==1:
                gradient=np.linspace(15,95,720)[None,:]
                image=np.clip(image*.48+gradient,0,255).astype(np.uint8)
            if index%3==2:
                image=np.clip(image.astype(float)+rng.normal(0,4,image.shape),0,255).astype(np.uint8)
            cases.append({'name':f'synthetic-{family}-{index:02}', 'dictionary':family,'image':image,
                          'sha256':hashlib.sha256(image.tobytes()).hexdigest()})
    # Blank targets and non-marker nested contours check spurious detections.
    for index in range(3):
        image=np.full((720,720),255,np.uint8)
        if index==1:
            for k in range(5):cv2.rectangle(image,(90+35*k,90+35*k),(630-35*k,630-35*k),index%2*0,5)
        if index==2:image=np.random.default_rng(12).integers(0,256,(720,720),dtype=np.uint8)
        cases.append({'name':f'negative-{index}','dictionary':'DICT_4X4_100','image':image,
                      'sha256':hashlib.sha256(image.tobytes()).hexdigest()})
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--corpus',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-sizes',type=int,nargs='+',default=[1,4,16])
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.build:build_library(args.library)
    cv2.setNumThreads(1)
    cases=synthetic()
    if args.corpus:
        for entry in json.loads(args.corpus.read_text()):
            path=Path(entry['path']);image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if image is None:raise ValueError(f'unreadable image: {path}')
            cases.append({'name':str(path),'dictionary':entry['dictionary'],'image':image,
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report={'scope':'identical grayscale images; excludes rendering; CPU reference without B1 special tracker',
            'opencv_version':cv2.__version__,'torch_version':torch.__version__,
            'device':torch.cuda.get_device_name(),
            'library_sha256':hashlib.sha256(args.library.read_bytes()).hexdigest(),
            'images':len(cases),'cases':[],'timings':[],
            'backend':'CUDA thresholds/contours/quad approximation/patch extraction; CPU grouping/ID/subpixel',
            'cpu_workers':args.workers}
    for family in dict.fromkeys(case['dictionary'] for case in cases):
        subset=[c for c in cases if c['dictionary']==family]
        d=cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,family));p=parameters()
        detector=GpuOpenCVDetector(d,p,str(args.library));cpu=cv2.aruco.ArucoDetector(d,p)
        images=np.stack([c['image'] for c in subset]);resident=torch.from_numpy(images).cuda()
        reference=[]
        for image in images:
            q,ids,_=cpu.detectMarkers(image)
            reference.append((q,[] if ids is None else ids.flatten().tolist()))
        detector.detect_gray(resident[:1])
        begin=time.perf_counter();rows,bytes_=detector.detect_gray(resident);elapsed=time.perf_counter()-begin
        for case,(q,ids),(rq,rids) in zip(subset,rows,reference):
            # Preserve duplicates and output order; set equality alone is insufficient.
            same_ids=ids==rids
            delta=None if not ids or not same_ids else float(max(np.max(np.abs(a-b)) for a,b in zip(q,rq)))
            report['cases'].append({'name':case['name'],'dictionary':family,'sha256':case['sha256'],
                                   'cpu_ids':rids,'gpu_ids':ids,'ordered_ids_equal':same_ids,
                                   'corner_max_delta_px':delta,
                                   'passed':same_ids and (delta is None or delta<=1e-3)})
        print(f'{family}: {len(subset)} images, {elapsed:.3f}s, {bytes_/images.nbytes:.3f} x grayscale transfer',flush=True)
        for batch in args.batch_sizes:
            # Representative native images where available, otherwise synthetic.
            indexes=[i for i,c in enumerate(subset) if not c['name'].startswith(('synthetic','negative'))]
            if not indexes:indexes=list(range(min(16,len(subset))))
            sample=resident[torch.tensor([indexes[i%len(indexes)] for i in range(batch)],device=resident.device)]
            detector.detect_gray(sample)
            with ThreadPoolExecutor(max_workers=min(args.workers,batch)) as pool:
                def cpu_image(image):
                    return cv2.aruco.ArucoDetector(d,p).detectMarkers(image)
                begin=time.perf_counter();host=sample.cpu().numpy();list(pool.map(cpu_image,host));cpu_s=time.perf_counter()-begin
                begin=time.perf_counter();_,amount=detector.detect_gray(sample);gpu_s=time.perf_counter()-begin
            report['timings'].append({'dictionary':family,'batch_size':batch,'gpu_frontend_wall_s':gpu_s,
                                      'cpu_download_and_detection_wall_s':cpu_s,'speedup_cpu_over_gpu':cpu_s/gpu_s,
                                      'gpu_download_bytes':amount,'cpu_grayscale_download_bytes':sample.numel()})
        detector.close()
    report['accuracy_gate_passed']=all(c['passed'] for c in report['cases'])
    report['max_corner_delta_px']=max((c['corner_max_delta_px'] or 0 for c in report['cases']),default=0)
    report['mismatched_images']=[c['name'] for c in report['cases'] if not c['passed']]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    if not report['accuracy_gate_passed']:raise SystemExit('reference compatibility gate failed')


if __name__=='__main__':main()
