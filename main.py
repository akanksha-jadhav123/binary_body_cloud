import cv2
import numpy as np
import time

from src.body_detector import BodyDetector
from src.pose_tracker import PoseTracker
from src.renderer import BinaryRenderer
from src.skeleton_renderer import SkeletonRenderer
from src.ui_overlay import UIOverlay

def draw_grid_background(width, height):
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(1, 15):
        y = height - int((i ** 1.5) * 3)
        if y > 0:
            cv2.line(bg, (0, y), (width, y), (30, 10, 0), 1)
    vanish_x, vanish_y = width // 2, height // 2
    for x in range(0, width + 100, 100):
        cv2.line(bg, (x, height), (vanish_x, vanish_y), (30, 10, 0), 1)
    return bg

def main():
    cap = cv2.VideoCapture(0)
    
    # 1. Lower native resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.setNumThreads(4)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2. THE MAGIC TRICK: Work at 90% size, display at 100%
    scale = 0.9 
    process_w = int(width * scale)
    process_h = int(height * scale)

    detector = BodyDetector()
    tracker = PoseTracker()
    # Initialize renderer with the smaller processing size
    renderer = BinaryRenderer(process_w, process_h, density=10) 
    skeleton = SkeletonRenderer()
    ui = UIOverlay(process_w, process_h)
    background = draw_grid_background(process_w, process_h)

    prev_time = time.time()
    print("Hybrid Binary Point Cloud started. Press 'q' to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        
        # Resize frame down for processing
        frame = cv2.resize(frame, (process_w, process_h))

        # --- LAYER 1: 2D Binary point cloud ---
        mask = detector.get_mask(frame)
        cloud = renderer.apply_effect(frame, mask)

        # --- LAYER 2: 3D skeleton ---
        landmarks = tracker.process(frame)
        skeleton_glow = np.zeros_like(frame)
        skeleton_glow = skeleton.draw(skeleton_glow, landmarks)
        skeleton_blur = cv2.GaussianBlur(skeleton_glow, (11, 11), 0)

        # --- COMPOSE ---
        composed = cv2.add(background.copy(), cloud)
        composed = cv2.add(composed, skeleton_glow)
        composed = cv2.add(composed, skeleton_blur)

        # --- FPS ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # --- UI ---
        composed = ui.draw(composed, tracker.get_keypoints(), fps, landmarks is not None)

        # 3. Scale the final composited image back up to window size
        final_display = cv2.resize(composed, (width, height), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Binary Body Point Cloud - Hybrid MoCap", final_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()