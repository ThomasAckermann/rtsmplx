import numpy as np


def get_lm_mapping():
    lm_mapping = np.array(
        [
            # Body Pose
            [0, 34],  # pelvis middle
            [1, 23],  # thigh right
            [2, 24],  # thigh left
            [3, 37],  # torso middle
            [4, 25],  # knee right
            [5, 26],  # knee left
            [6, 38],  # torso second top
            [7, 29],  # back foot right
            [8, 30],  # back foot left
            [9, 39],  # torso top
            [10, 31],  # front foot right
            [11, 32],  # front foot left
            [12, 33],  # neck
            [13, 35],  # inner shoulder right
            [14, 36],  # inner shoulder left
            [15, 0],  # mouth right
            [16, 11],  # shoulder right
            [17, 12],  # shoulder left
            [18, 13],  # arm right
            [19, 14],  # arm left
            [20, 19],  # hand right
            [21, 20],  # hand left
            [22, 10],  # mouth left
            [23, 2],  # right eye
            [24, 4],  # left eye
        ]
    )

    face_mapping = np.array(
        [np.arange(start=76, stop=127), np.arange(start=40, stop=91)]
    ).T
    lm_mapping = np.concatenate((lm_mapping, face_mapping), axis=0)

    return lm_mapping


if __name__ == "__main__":
    print(get_lm_mapping())
