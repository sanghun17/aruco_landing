import unittest
import numpy as np
from aruco_landing.pose_transition import PoseTransition


class TestTransition(unittest.TestCase):
    def ready_router(self, allow=True):
        r=PoseTransition(allow=allow)
        for i in range(150):
            t=100+i*.01
            for key in ('aligned','visible','inliers'): r.set_quality(key,True,t)
            r.ingest('optitrack',t,np.eye(4),t)
            if i>5: r.ingest('marker',t-.05,np.eye(4),t)
        return r,t

    def test_qualified_switch_and_monotonic_output(self):
        r,t=self.ready_router()
        self.assertTrue(r.select(True,t)[0])
        self.assertFalse(r.ingest('marker',t-.04,np.eye(4),t+.01))
        self.assertTrue(r.ingest('marker',t+.01,np.eye(4),t+.06))
        self.assertFalse(r.ingest('optitrack',t+.07,np.eye(4),t+.07))

    def test_switch_lock(self):
        r,t=self.ready_router(False)
        self.assertFalse(r.select(True,t)[0])

    def test_stale_stamps_are_rejected_even_if_just_received(self):
        r,t=self.ready_router()
        self.assertFalse(r.ingest('marker',t-.3,np.eye(4),t))
        self.assertFalse(r.select(True,t+.3)[0])

    def test_large_jump_restarts_qualification(self):
        r,t=self.ready_router(); bad=np.eye(4);bad[0,3]=1
        r.ingest('marker',t-.04,bad,t+.01)
        self.assertFalse(r.select(True,t+.01)[0])
        self.assertEqual(r.good_count,0)

    def test_marker_continues_without_mocap_but_stops_for_lost_quality(self):
        r,t=self.ready_router();self.assertTrue(r.select(True,t)[0])
        for key in ('visible','inliers'):r.set_quality(key,True,t+1)
        self.assertTrue(r.ingest('marker',t+1,np.eye(4),t+1))
        r.set_quality('visible',False,t+1.01)
        self.assertFalse(r.ingest('marker',t+1.02,np.eye(4),t+1.02))
        self.assertEqual(r.source,'marker')
        self.assertFalse(r.select(False,t+1.02)[0])

    def test_receipt_and_measurement_gaps_restart_qualification(self):
        r,t=self.ready_router()
        for key in ('aligned','visible','inliers'):r.set_quality(key,True,t+1)
        r.ingest('optitrack',t+1,np.eye(4),t+1)
        r.ingest('marker',t+1,np.eye(4),t+1)
        self.assertFalse(r.select(True,t+1)[0])

    def test_half_second_fallback_requires_fresh_optitrack(self):
        r,t=self.ready_router();r.fallback_timeout=.5
        self.assertTrue(r.select(True,t)[0])
        r.set_quality('visible',False,t+.01)
        r.ingest('optitrack',t+.49,np.eye(4),t+.49)
        self.assertIsNone(r.fallback(t+.49))
        self.assertIsNone(r.fallback(t+.8)) # stale mocap must not be selected
        r.ingest('optitrack',t+.81,np.eye(4),t+.81)
        self.assertIsNotNone(r.fallback(t+.81))
        self.assertEqual(r.source,'optitrack')
        self.assertEqual(r.good_count,0)

    def test_repeated_requalification_after_timeout(self):
        r,t=self.ready_router();r.fallback_timeout=.5
        for cycle in range(3):
            self.assertTrue(r.select(True,t)[0])
            r.ingest('optitrack',t+.51,np.eye(4),t+.51)
            self.assertIsNotNone(r.fallback(t+.51))
            for i in range(150):
                now=t+.6+i*.01
                for key in ('aligned','visible','inliers'):r.set_quality(key,True,now)
                r.ingest('optitrack',now,np.eye(4),now)
                r.ingest('marker',now,np.eye(4),now)
            t=now

if __name__=='__main__':unittest.main()


class IntegratedTransitionTest(TestTransition):
    def test_trial_permission_resets_preflight_dwell(self):
        r,t=self.ready_router();r.set_enabled(False)
        self.assertFalse(r.select(True,t)[0])
        r.set_enabled(True)
        self.assertEqual(r.good_count,0)
        self.assertFalse(r.select(True,t)[0])

    def test_selected_marker_outlier_is_not_published_or_counted_as_valid(self):
        r,t=self.ready_router();r.reject_inconsistent_marker=True
        self.assertTrue(r.select(True,t)[0]);last=r.last_valid_marker_receipt
        bad=np.eye(4);bad[0,3]=.3
        for key in ('visible','inliers'):r.set_quality(key,True,t+.1)
        r.ingest('optitrack',t+.1,np.eye(4),t+.1)
        self.assertFalse(r.ingest('marker',t+.1,bad,t+.1))
        self.assertEqual(r.last_valid_marker_receipt,last)

    def test_fallback_latches_and_requires_new_trial_reset(self):
        r,t=self.ready_router();r.latch_fallback=True;r.fallback_timeout=.5
        self.assertTrue(r.select(True,t)[0])
        r.ingest('optitrack',t+.51,np.eye(4),t+.51)
        self.assertIsNotNone(r.fallback(t+.51));self.assertTrue(r.fallback_latched)
        for i in range(150):
            now=t+.6+i*.01
            for key in ('aligned','visible','inliers'):r.set_quality(key,True,now)
            r.ingest('optitrack',now,np.eye(4),now);r.ingest('marker',now,np.eye(4),now)
        self.assertFalse(r.select(True,now)[0])

    def test_disabled_marker_cannot_leak_without_fresh_mocap(self):
        r,t=self.ready_router();self.assertTrue(r.select(True,t)[0]);r.set_enabled(False)
        self.assertFalse(r.return_to_mocap(t+1,latch=True))
        for key in ('visible','inliers'):r.set_quality(key,True,t+1)
        self.assertFalse(r.ingest('marker',t+1,np.eye(4),t+1))
