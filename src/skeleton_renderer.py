import cv2
import numpy as np

class SkeletonRenderer:
    def __init__(self):
        self.connections = [
            (0,1),(1,2),(2,3),(3,7),        # face right
            (0,4),(4,5),(5,6),(6,8),        # face left
            (9,10),                          # mouth
            (11,12),(11,13),(13,15),        # left arm
            (12,14),(14,16),                # right arm
            (11,23),(12,24),(23,24),        # torso
            (23,25),(25,27),(27,29),(29,31),# left leg
            (24,26),(26,28),(28,30),(30,32) # right leg
        ]

    def _project(self, lm, depth_scale=0.15):
        perspective = 1.0 - (lm["z"] * depth_scale)
        perspective = max(0.5, min(1.5, perspective))
        return int(lm["x"]), int(lm["y"]), perspective

    def draw(self, canvas, landmarks, color=(255, 200, 0)):
        if not landmarks:
            return canvas

        for a, b in self.connections:
            lmA, lmB = landmarks[a], landmarks[b]
            if lmA["vis"] < 0.4 or lmB["vis"] < 0.4:
                continue
            
            x1, y1, p1 = self._project(lmA)
            x2, y2, p2 = self._project(lmB)
            thickness = max(1, int(2 * min(p1, p2))) # Thinner lines

            # Draw just ONE thin sharp line (the blur in main.py handles the glow)
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness) 

        for lm in landmarks:
            if lm["vis"] < 0.4:
                continue
            x, y, p = self._project(lm)
            radius = max(2, int(3 * p)) # Smaller dots
            cv2.circle(canvas, (x, y), radius, (255, 255, 255), -1)

        return canvas