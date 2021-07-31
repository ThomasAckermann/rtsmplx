import face_alignment
import mediapipe as mp
import cv2


class Landmarks:
    """
    Landmark class is used to combine landmarks for body, head and hands
    """
    def __init__(self, image, head=False, hands=False, debug=False):
        self.image = image.numpy()
        self.face_lm = None
        self.hand_lm = None
        self.body_lm = self.body_landmarks(debug=False)
        if head == True:
            self.face_lm = self.face_landmarks()
        if hands == True:
            self.hand_lm = self.hand_landmarks()
        
    def face_landmarks(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D)
        prediction = fa.get_landmarks(self.image)
        return prediction

    def hand_landmarks(self):
        drawingModule = mp.solutions.drawing_utils
        handsModule = mp.solutions.hands
        with handsModule.Hands(static_image_mode=True) as hands:
            results = hands.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(self.image, handLandmarks, handsModule.HAND_CONNECTIONS)
        return results.multi_hand_landmarks
    
    def body_landmarks(self, debug=False):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
            results = pose.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        return results.pose_landmarks

