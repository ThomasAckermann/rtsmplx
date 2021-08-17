import numpy as np


def get_lm_mapping():
    lm_mapping = np.array(
        [
            # [0,], # pelvis middle
            [1, 23],  # pelvis right
            [2, 24],  # pelvis left
            # [3,], # torso middle
            [4, 25],  # knee right
            [5, 26],  # knee left
            # [6,], # torso second top
            [7, 29],  # back foot right
            [8, 30],  # back foot left
            # [9,], # torso top
            [10, 31],  # front foot right
            [11, 32],  # front foot left
            [12, 9],  # neck
            # [13,], # inner shoulder right
            # [14,], # inner shoulder left
            # [15,], # nose right
            [16, 11],  # should right
            [17, 12],  # shoulder left
            [18, 13],  # arm right
            [19, 14],  # arm left
            [20, 19],  # hand right
            [21, 20],  # hand left
            [22, 0],  # nose left
            [23, 2],  # right eye
            [24, 4],  # left eye
        ]
    )

    """
    15,16 wrist
    17 18 hand top
    19 20 hand middle
    21 22 hand middle
    """
    return lm_mapping
