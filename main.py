import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.setNumThreads(4)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    scale = 0.9 
    process_w = int(width * scale)
    process_h = int(height * scale)

    detector = BodyDetector()
    tracker = PoseTracker()
    renderer = BinaryRenderer(process_w, process_h, density=10) 
    skeleton = SkeletonRenderer()
    ui = UIOverlay(process_w, process_h)
    background = draw_grid_background(process_w, process_h)

    # --- RECORDING SETUP ---
    is_recording = False
    video_writer = None
    csv_file = None
    csv_writer = None
    
    os.makedirs("output", exist_ok=True) # Create output folder

    prev_time = time.time()
    print("Started. Press 'R' to Record Video, 'S' to Save Data, 'q' to Quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
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

        # Display recording indicator
        if is_recording:
            cv2.circle(composed, (process_w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(composed, "REC", (process_w - 80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Scale up for display
        final_display = cv2.resize(composed, (width, height), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Binary Body Point Cloud - Hybrid MoCap", final_display)

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('r'):
            if not is_recording:
                # Start Recording
                is_recording = True
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"output/mocap_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))
                print(f"Started recording video: {video_filename}")
            else:
                # Stop Recording
                is_recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print("Stopped recording video.")

        elif key == ord('s'):
            # Save Data to CSV
            if landmarks:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"output/mocap_data_{timestamp}.csv"
                with open(csv_filename, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    # Write headers
                    headers = ["Joint", "X", "Y", "Z", "Visibility"]
                    writer.writerow(headers)
                    # Write 33 landmarks
                    for i, lm in enumerate(landmarks):
                        writer.writerow([i, lm['x'], lm['y'], lm['z'], lm['vis']])
                print(f"Saved MoCap data to: {csv_filename}")

        # Write frame to video if recording
        if is_recording and video_writer is not None:
            video_writer.write(final_display)

    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()