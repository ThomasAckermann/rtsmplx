import numpy as np


def get_lm_mapping():
    lm_mapping = np.array(
        [
            # Body Pose
            # [0, 34],  # pelvis middle
            [1, 23],  # thigh right
            [2, 24],  # thigh left
            # [3, 37],  # torso middle
            [4, 25],  # knee right
            [5, 26],  # knee left
            # [6, 38],  # torso second top
            [7, 29],  # back foot right
            [8, 30],  # back foot left
            # [9, 39],  # torso top
            [10, 31],  # front foot right
            [11, 32],  # front foot left
            # [12, 33],  # neck
            # [13, 35],  # inner shoulder right
            # [14, 36],  # inner shoulder left
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
    hand_mapping = np.array(
            [
                # left hand
                # 0, # wrist
                [1, 37], # thumb_cmc
                # 2,  # thumb_mcp
                [3, 38], # thumb_ip
                [4, 39], # thumb_tip
                [5, 25], # index_finger_mcp
                # 6 # index_finger_pip
                [7, 27], # index_finger_dip
                [8, 26], # index_finger_tip
                [9, 28], # middle_finder_mcp
                # 10 # middle_finger_pip
                [11, 30], # middle_finger_dip
                [12, 29], # middle_finger_tip
                [13, 34], # ring_finger_mcp
                # 14 # ring_finger_pip
                [15, 36], # ring_finger_dip
                [16, 35], # ring_finger_tip
                [17, 31], # pinky_mcp
                # 18 # pinky_pip
                [19, 33], # pinky_dip
                [20, 32], # pinky_tip

                # right hand
                # 0, # wrist
                [1, 37+15], # thumb_cmc
                # 2,  # thumb_mcp
                [3, 38+15], # thumb_ip
                [4, 39+15], # thumb_tip
                [5, 25+15], # index_finger_mcp
                # 6 # index_finger_pip
                [7, 27+15], # index_finger_dip
                [8, 26+15], # index_finger_tip
                [9, 28+15], # middle_finder_mcp
                # 10 # middle_finger_pip
                [11, 30+15], # middle_finger_dip
                [12, 29+15], # middle_finger_tip
                [13, 34+15], # ring_finger_mcp
                # 14 # ring_finger_pip
                [15, 36+15], # ring_finger_dip
                [16, 35+15], # ring_finger_tip
                [17, 31+15], # pinky_mcp
                # 18 # pinky_pip
                [19, 33+15], # pinky_dip
                [20, 32+15], # pinky_tip
                ]
            )
    """

    face_mapping = np.array(
        [np.arange(start=76, stop=127), np.arange(start=40, stop=91)]
    ).T
    lm_mapping = np.concatenate((lm_mapping, face_mapping), axis=0)
    """

    return lm_mapping


if __name__ == "__main__":
    print(get_lm_mapping())


