import cv2
import numpy as np
import random

class BinaryRenderer:
    def __init__(self, width, height, density=10):
        self.width = width
        self.height = height
        self.density = density
        self.frame_count = 0
        self.binary_texture = self._generate_binary_texture()

    def _generate_binary_texture(self):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        color = (255, 255, 255)

        for y in range(0, self.height, self.density):
            for x in range(0, self.width, self.density):
                char = "1" if random.random() > 0.5 else "0"
                cv2.putText(canvas, char, (x, y), font, font_scale,
                            color, thickness, 1)
        return canvas

    def apply_effect(self, frame, mask):
        # Flicker effect: Regenerate texture every 5 frames
        self.frame_count += 1
        if self.frame_count % 5 == 0:
            self.binary_texture = self._generate_binary_texture()

        output = np.zeros_like(frame)

        color_layer = np.zeros_like(self.binary_texture)
        color_layer[:] = (255, 200, 0)  # Cyan / Electric Blue

        colored_text = cv2.bitwise_and(color_layer, self.binary_texture)

        if mask.shape[:2] != colored_text.shape[:2]:
            mask = cv2.resize(mask, (colored_text.shape[1], colored_text.shape[0]))

        masked_text = cv2.bitwise_and(colored_text, colored_text, mask=mask)

        glow = cv2.GaussianBlur(masked_text, (21, 21), 0)
        final_effect = cv2.addWeighted(masked_text, 1.0, glow, 0.8, 0)

        output = cv2.add(output, final_effect)
        return output