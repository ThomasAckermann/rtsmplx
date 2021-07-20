import face_alignment
import mediapipe as mp
import cv2


class Landmarks:
    def __init__(self, image, head=False, hands=False):
        self.image = image
        self.face_lm = None
        self.hand_lm = None
        self.body_lm = self.body_landmarks()
        if head == True:
            self.face_lm = self.face_landmarks()
        if hands == True:
            self.hand_lm = self.hand_landmarks()
        

    def face_landmarks(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.__3D)
        prediction = fa.get_landmarks(self.image)
        return prediction

    def hand_landmarks(self):
        pass
    
    def body_landmarks(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
            results = pose.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            # Draw pose landmarks on the image.
            annotated_image = self.image.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
            # Plot pose world landmarks.
            mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)        
        return results.pose_landmarks


