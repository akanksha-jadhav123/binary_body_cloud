import cv2
import mediapipe as mp
import numpy as np

class BodyDetector:
    def __init__(self):
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=0) # model_selection=0 is general (faster)

    def get_mask(self, frame):
        # TRICK: Resize down to 320x240 just for the AI processing
        small_frame = cv2.resize(frame, (320, 240))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        results = self.segmenter.process(rgb_frame)
        
        # Get mask and threshold it
        mask = np.where(results.segmentation_mask > 0.1, 255, 0).astype(np.uint8)
        
        # TRICK: Resize the mask back to the original frame size
        h, w = frame.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask