import unittest
import cv2
import numpy as np
from aruco_landing.physical_pad import PhysicalPadDetector,inverse
from aruco_landing.nested_apriltag import NestedAprilTagTracker


class NestedAprilTagTest(unittest.TestCase):
    def test_pose_follows_image_motion_and_rejects_missing_target(self):
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        children=[6,20,7,29,15,21,12,24,9]
        canvas=np.rot90(cv2.aruco.generateImageMarker(dictionary,166,560),2).copy()
        markers=[dict(id=166,side_m=.7,center_m=dict(x=0,y=0),yaw_deg=180)]
        for i,mid in enumerate(children):
            x,y=[0,245,490][i%3],[0,245,490][i//3]
            canvas[y:y+70,x:x+70]=np.rot90(cv2.aruco.generateImageMarker(dictionary,mid,70),2)
            markers.append(dict(id=mid,side_m=.0875,center_m=dict(x=(x+35-280)/800,y=(280-y-35)/800),yaw_deg=180))
        estimator=PhysicalPadDetector(dict(dictionary='DICT_APRILTAG_36h11',markers=markers))
        tracker=NestedAprilTagTracker(estimator,children)
        K=np.array([[708.,0,360.],[0,708.,360.],[0,0,1.]])
        D=np.zeros(5)
        camera_R=np.diag([1.,-1.,-1.])
        pad_corners=np.array([[-.35,.35,0],[.35,.35,0],[.35,-.35,0],[-.35,-.35,0]])
        first=None
        for frame in range(6):
            position=np.array([.06+.002*frame,-.08-.001*frame,1.8-.008*frame])
            T=np.eye(4);T[:3,:3]=camera_R;T[:3,3]=-camera_R@position
            q=cv2.projectPoints(pad_corners,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],K,D)[0].reshape(4,2)
            H=cv2.getPerspectiveTransform(np.float32([[-.5,-.5],[559.5,-.5],[559.5,559.5],[-.5,559.5]]),q.astype(np.float32))
            image=cv2.warpPerspective(canvas,H,(720,720),borderValue=255)
            corners,ids=tracker.detect(image,K,D)
            pose=estimator.estimate(corners,ids,K,D)
            self.assertIsNotNone(pose,frame)
            tracker.update(pose)
            estimated=inverse(pose['camera_from_pad'])[:3,3]
            self.assertLess(np.linalg.norm(estimated-position),.03)
            if first is None:first=estimated
        self.assertGreater(np.linalg.norm(estimated-first),.03)
        corners,ids=tracker.detect(np.full((720,720),255,np.uint8),K,D)
        self.assertEqual(ids,[])
        self.assertIsNone(estimator.estimate(corners,ids,K,D))
        tracker.reset()
        self.assertIsNone(tracker.previous)


if __name__=='__main__':unittest.main()
