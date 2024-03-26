import cv2

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None