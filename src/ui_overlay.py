import cv2
import numpy as np

class UIOverlay:
    def __init__(self, width, height):
        self.w = width
        self.h = height

    def _text(self, img, txt, pos, scale=0.4, color=(255, 200, 0), thick=1):
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    def draw(self, frame, keypoints, fps, tracked):
        w, h = self.w, self.h
        
        # Create a copy of the frame to draw a semi-transparent dark UI backdrop
        overlay = frame.copy()
        
        # Draw dark panels behind the UI text for readability
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)              # Top bar
        cv2.rectangle(overlay, (0, h - 80), (250, h), (0, 0, 0), -1)        # Bottom left
        cv2.rectangle(overlay, (w - 300, h - 50), (w, h), (0, 0, 0), -1)    # Bottom right
        
        # Blend the overlay with the original frame (0.7 opacity)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # --- TOP LEFT: FACE POINT CLOUD BOX ---
        self._text(frame, "FACE CLOUD (0/1)", (15, 20))
        cv2.rectangle(frame, (15, 30), (135, 110), (255, 200, 0), 1)
        for i in range(20, 130, 8):
            for j in range(35, 105, 8):
                char = "1" if (i+j) % 3 == 0 else "0"
                cv2.putText(frame, char, (i, j), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 200, 0), 1)
        
        status = "1" if tracked else "0"
        self._text(frame, f"STATUS: {status}", (15, 105), 0.35, (255, 255, 255))

        # --- TOP RIGHT: ARM POINT CLOUD BOX ---
        self._text(frame, "ARM CLOUD (0/1)", (w - 150, 20))
        cv2.rectangle(frame, (w - 150, 30), (w - 15, 110), (255, 200, 0), 1)
        for i in range(w - 145, w - 25, 8):
            for j in range(35, 105, 8):
                char = "0" if (i+j) % 3 == 0 else "1"
                cv2.putText(frame, char, (i, j), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 200, 0), 1)
                
        self._text(frame, f"STATUS: {status}", (w - 145, 105), 0.35, (255, 255, 255))

        # --- BOTTOM LEFT: LEGEND + COORDINATES ---
        self._text(frame, "1 = Point detected", (15, h - 60), 0.4, (255, 255, 255))
        self._text(frame, "0 = No point", (15, h - 45), 0.4, (255, 255, 255))

        if keypoints:
            head = keypoints["HEAD"]
            lh = keypoints["LEFT_HAND"]
            self._text(frame, f"HEAD    X:{int(head['x'])} Y:{int(head['y'])} Z:{head['z']:.2f}", (15, h - 25), 0.35)
            self._text(frame, f"L-HAND  X:{int(lh['x'])} Y:{int(lh['y'])} Z:{lh['z']:.2f}", (15, h - 5), 0.35)

        # --- TOP CENTER: FPS ---
        self._text(frame, f"FPS: {int(fps)}", (w // 2 - 30, 25), 0.6, (0, 255, 0), 2)

        # --- BOTTOM RIGHT: TITLE ---
        self._text(frame, "BINARY BODY POINT CLOUD", (w - 250, h - 20), 0.45, (255, 255, 255))

        return frame