import face_alignment
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import torch


class Landmarks:
    """combines landmarks for body, head and hands

    Keyword arguments:
    image   --  image in torch tensor format
    head    --  bool that decides if head landmarks are generated (default: false)
    hand    --  bool that decides if hand landmarks are generated (default: false)
    """

    def __init__(self, image, head=False, hands=False):
        super(Landmarks, self).__init__()
        self.image = image
        self.image_shape = self.image.size()
        self.image_height = self.image_shape[0]
        self.image_width = self.image_shape[1]
        self.image_channels = self.image_shape[2]
        self.body_lm = self.body_landmarks()
        self.use_head_lms = head
        self.use_hands_lms = hands
        if self.use_head_lms == True:
            self.face_lm = self.face_landmarks()
        else:
            self.face_lm = None
        if self.use_hands_lms == True:
            self.hand_lm = self.hand_landmarks()
        else:
            self.hand_lm = None

    def face_landmarks(self):
        image_face = self.image.reshape(
            self.image_height, self.image_width, self.image_channels
        )
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False
        )
        prediction = torch.tensor(fa.get_landmarks(image_face)[0])[:, :2]
        prediction[:, 0] = prediction[:, 0] / self.image_height
        prediction[:, 1] = prediction[:, 1] / self.image_width
        return prediction

    def hand_landmarks(self):
        image_hands = self.image.numpy()
        drawingModule = mp.solutions.drawing_utils
        handsModule = mp.solutions.hands
        with handsModule.Hands(static_image_mode=True) as hands:
            results = hands.process(cv2.cvtColor(image_hands, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(
                        image_hands, handLandmarks, handsModule.HAND_CONNECTIONS
                    )
        return results

    def body_landmarks(self):
        mp_pose = mp.solutions.pose
        image_body = self.image.numpy()
        image_body = cv2.cvtColor(image_body, cv2.COLOR_BGR2RGB)
        with mp_pose.Pose(
            static_image_mode=True, model_complexity=2, min_detection_confidence=0.5
        ) as pose:
            results = pose.process(image_body)
        results = torch.Tensor([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
        body_lms = torch.zeros(40, 2)
        body_lms[:33, :] = results
        body_lms[33] = (results[11] + results[12]) / 2  # torso top
        body_lms[34] = (results[23] + results[24]) / 2  # torso bottom
        body_lms[35] = (body_lms[33] + results[11]) / 2  # inner shoulder right
        body_lms[36] = (body_lms[33] + results[12]) / 2  # inner shoulder left
        body_axis = body_lms[33] - body_lms[34]
        body_lms[37] = body_lms[34] + (1 / 4) * body_axis
        body_lms[38] = body_lms[34] + (2 / 4) * body_axis
        body_lms[39] = body_lms[34] + (3 / 4) * body_axis

        return body_lms

    def plot_landmarks(self):
        plt.scatter(
            self.body_lm[:, 0] * self.image_height,
            self.body_lm[:, 1] * self.image_width,
        )

        if self.use_head_lms == True:
            plt.scatter(
                self.face_lm[:, 0] * self.image_height,
                self.face_lm[:, 1] * self.image_width,
            )
        if self.use_head_lms == True:
            plt.scatter(
                self.hand_lm[:, 0] * self.image_height,
                self.hand_lm[:, 1] * self.image_width,
            )
        plt.imshow(self.image)
        plt.show()

        return "Done"
