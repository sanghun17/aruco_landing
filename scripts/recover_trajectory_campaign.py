#!/usr/bin/env python3
"""Recover pad-scoped progress from finalized bags; requires disarmed pilot mode."""
import argparse
import json
from pathlib import Path
import rospy
from mavros_msgs.msg import State,ExtendedState
from aruco_landing.campaign import flight_record,recover_progress,acquire_campaign_lock
from aruco_landing.trajectory_set import TrajectorySet


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag-dir',type=Path,required=True)
    p.add_argument('--progress-root',type=Path,required=True)
    p.add_argument('--set-file',type=Path,required=True)
    args=p.parse_args()
    rospy.init_node('recover_trajectory_campaign',anonymous=True)
    lease=acquire_campaign_lock(args.progress_root)
    nodes=rospy.get_master().getSystemState()[2]
    if any('/landing_trial' in owners for group in nodes for _,owners in group):
        raise RuntimeError('stop the planner before recovering progress')
    state=rospy.wait_for_message('/mavros/state',State,timeout=4.)
    ext=rospy.wait_for_message('/mavros/extended_state',ExtendedState,timeout=4.)
    if (not state.connected or state.armed or state.mode!='POSCTL'
            or ext.landed_state!=ExtendedState.LANDED_STATE_ON_GROUND
            or state.header.stamp.is_zero() or ext.header.stamp.is_zero()
            or not 0<=rospy.get_time()-state.header.stamp.to_sec()<2.
            or not 0<=rospy.get_time()-ext.header.stamp.to_sec()<2.):
        raise RuntimeError('fresh connected disarmed ground POSCTL required; no changes made')
    owner=Path(__file__).resolve().parents[1];records=[];skipped=[]
    for bag in sorted(args.bag_dir.glob('*.bag')):
        try:records.append(flight_record(bag,owner))
        except Exception as error:skipped.append(dict(bag=bag.name,reason=str(error)))
    if not records:raise RuntimeError('no usable finalized trajectory recordings; progress was not changed: '+str(skipped))
    state=rospy.wait_for_message('/mavros/state',State,timeout=4.)
    if not state.connected or state.armed or state.mode!='POSCTL':raise RuntimeError('pilot state changed during recovery')
    result=recover_progress(TrajectorySet(args.set_file),args.progress_root,records)
    print(json.dumps(dict(campaigns=result,skipped=skipped),indent=2))

if __name__=='__main__':main()
