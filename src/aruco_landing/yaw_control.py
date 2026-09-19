"""Bounded pad-frame body-heading feedback (radians, positive about pad +Z)."""
import math


def yaw_feedback(quaternion, target=0., kp=1., max_rate=.35, deadband=math.radians(1.)):
    values=list(quaternion)
    if len(values)!=4 or not all(math.isfinite(x) for x in values+[target,kp,max_rate,deadband]):
        raise ValueError('yaw feedback requires finite quaternion and gains')
    if kp<0 or max_rate<=0 or not 0<=deadband<math.pi:
        raise ValueError('invalid yaw gain, rate limit, or deadband')
    norm=math.sqrt(sum(x*x for x in values))
    if norm<1e-8:raise ValueError('zero heading quaternion')
    x,y,z,w=[v/norm for v in values]
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    error=math.atan2(math.sin(target-yaw),math.cos(target-yaw))
    rate=0. if abs(error)<=deadband else max(-max_rate,min(max_rate,kp*error))
    return error,rate
