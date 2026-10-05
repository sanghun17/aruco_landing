"""Optical template tracking for the reconstructed nested AprilTag pad.

This is a compatible observation frontend, not a reproduction of MVFAN's
published detector. OpenCV acquires an ordinary AprilTag pose. Subsequent
frames rectify small-tag ROIs from the previous optical estimate, align their
intensity patterns, and verify their decoded IDs before reporting corners.
No simulator poses or corners enter this class. PhysicalPadDetector.estimate
remains the common PnP estimator.
"""
import cv2
import numpy as np


class NestedAprilTagTracker:
    def __init__(self, estimator, child_ids):
        self.estimator = estimator
        self.child_ids = tuple(child_ids)
        self.previous = None
        self.size = 64
        n = self.size
        self.canonical = np.float32([[-.5,-.5],[n-.5,-.5],[n-.5,n-.5],[-.5,n-.5]])
        self.templates = {mid:cv2.aruco.generateImageMarker(estimator.dictionary,mid,n).astype(np.float32)/255
                          for mid in self.child_ids}
        self.criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,40,1e-5)
        samples = ((np.arange(6)+1.5)*n/8-.5).astype(int)
        self.samples = np.ix_(samples,samples)

    def reset(self):
        self.previous = None

    def update(self, observation):
        if observation is not None:
            self.previous = observation['camera_from_pad'].copy()

    def detect(self, gray, K, D):
        corners, ids = [], []
        if self.previous is not None:
            rv = cv2.Rodrigues(self.previous[:3,:3])[0]
            tv = self.previous[:3,3]
            height,width = gray.shape
            for mid in self.child_ids:
                q = cv2.projectPoints(self.estimator.models[mid],rv,tv,K,D)[0].reshape(4,2).astype(np.float32)
                edge = np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1).mean()
                if edge<12 or not np.isfinite(q).all() or q.min()<1 or q[:,0].max()>width-2 or q[:,1].max()>height-2:
                    continue
                H = cv2.getPerspectiveTransform(self.canonical,q)
                try:
                    rectified = cv2.warpPerspective(gray,np.linalg.inv(H),(self.size,self.size),flags=cv2.INTER_LINEAR).astype(np.float32)/255
                    # Inter-frame residual motion is locally affine; rectification
                    # already accounts for the estimated projective geometry.
                    score,W = cv2.findTransformECC(self.templates[mid],rectified,np.eye(2,3,dtype=np.float32),
                        cv2.MOTION_AFFINE,self.criteria,None,3)
                    W = np.vstack([W,[0,0,1]])
                    refined = cv2.perspectiveTransform(self.canonical[None],H@W).reshape(4,2)
                    aligned = cv2.warpPerspective(rectified,W,(self.size,self.size),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
                    bits = (aligned[self.samples]>.5).astype(np.uint8)
                    decoded,found,rotation = self.estimator.dictionary.identify(bits,.4)
                except (cv2.error,np.linalg.LinAlgError):
                    continue
                if (score<.9 or not decoded or found!=mid or rotation!=0 or not np.isfinite(refined).all()
                        or np.linalg.norm(refined-q,axis=1).max()>.25*edge+2
                        or not cv2.isContourConvex(refined.reshape(4,1,2))):
                    continue
                corners.append(refined.reshape(1,4,2));ids.append(mid)
        if not ids:
            # Acquisition/reacquisition is optical dictionary detection; the
            # controller receives a missing pose if this also fails.
            corners,found,_ = self.estimator.detector.detectMarkers(gray)
            ids = [] if found is None else found.flatten().tolist()
        return corners,ids
