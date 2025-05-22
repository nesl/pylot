import numpy as np

import params

def radians_to_steer(rad: float, steer_gain: float):
    """Converts radians to steer input.

    Returns:
        :obj:`float`: Between [-1.0, 1.0].
    """
    steer = steer_gain * rad
    if steer > 0:
        steer = min(steer, 1)
    else:
        steer = max(steer, -1)
    return steer


def steer_to_radians(steer: float, steer_gain: float):
    """Converts radians to steer input.

    Assumes max steering angle is -45, 45 degrees.

    Returns:
        :obj:`float`: Steering in radians.
    """
    rad = steer / steer_gain
    if rad > 0:
        rad = min(rad, np.pi / 2)
    else:
        rad = max(rad, -np.pi / 2)
    return rad

def as_numpy_array_2D(self):
    wx = []
    wy = []
    for wp in self.waypoints:
        wx.append(wp.location.x)
        wy.append(wp.location.y)
    return np.array([wx, wy])
