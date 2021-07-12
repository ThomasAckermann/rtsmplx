import cv2
import math
import smplx

def video_capture(path, framerate=5):
    video_capture = cv2.VideoCapture(path)
    framerate = video_capture.get(framerate)
    count = 1
    while (video_capture.isOpened()):
        frame_id = video_capture.get(1)
        ret, frame = video_capture.read()
        if (ret != True):
            break
        if (frame_id % math.floor(framerate) == 0):
            filename = "frame%d.jpg" % count; count+=1
            cv2.imwrite(filename, frame)
    video_capture.release()
    return "Done"
    

def create_model(path):
    model = smplx.body_models.create(path)
    


