"""
Some functions that will probably be used in different places,
such as euler/direction vector conversion.
"""
__author__ = 'Matjaz'

import numpy as np


def zyx_euler_to_rotation_matrix(th):
    """Convert the ZYX order (the one LMS uses) Euler
        angles to rotation matrix. Angles are given
        in radians.

        Note:
            Actually Tait-Bryant angles.
        """
    # -- Calculate sine and cosine values first.
    sz, sy, sx = [np.sin(value) for value in th]
    cz, cy, cx = [np.cos(value) for value in th]

    # -- Create and populate the rotation matrix.
    rotation_matrix = np.zeros((3, 3), dtype=float)

    rotation_matrix[0, 0] = cy*cz
    rotation_matrix[0, 1] = cz*sx*sy - cx*sz
    rotation_matrix[0, 2] = cx*cz*sy + sx*sz
    rotation_matrix[1, 0] = cy*sz
    rotation_matrix[1, 1] = cx*cz + sx*sy*sz
    rotation_matrix[1, 2] = -cz*sx + cx*sy*sz
    rotation_matrix[2, 0] = -sy
    rotation_matrix[2, 1] = cy*sx
    rotation_matrix[2, 2] = cx*cy

    return rotation_matrix