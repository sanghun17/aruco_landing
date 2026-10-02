"""Numerical landing policy shared by ROS and lockstep batch evaluation."""
import math


def limit_xy(x, y, limit):
    speed = math.hypot(x, y)
    scale = min(1.0, limit / speed) if speed else 1.0
    return x * scale, y * scale


def filtered_derivative(error, previous, derivative, dt, tau, limit):
    # Preserve the ROS controller's behavior after stale/duplicate samples.
    if previous is None or not 0.0 < dt <= 0.25:
        return derivative
    alpha = 1.0 if tau == 0.0 else dt / (tau + dt)
    return limit_xy(*[(1-alpha)*d + alpha*(e-p)/dt
                      for e, p, d in zip(error, previous, derivative)], limit)


def horizontal_feedback(position, derivative, age, kp, kd, limit):
    return limit_xy(*[kp*(-p+d*age) + kd*d
                      for p, d in zip(position, derivative)], limit)


def predicted_height(height, age, descent_speed, lead):
    return height - descent_speed * (age + lead)
