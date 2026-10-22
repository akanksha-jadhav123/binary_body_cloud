import cv2
import mediapipe as mp

class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # Changed to 0 for faster real-time performance on CPU
        self.pose = self.mp_pose.Pose(
            model_complexity=0,       
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarks = None

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            h, w = frame.shape[:2]
            self.landmarks = []
            for lm in results.pose_landmarks.landmark:
                self.landmarks.append({
                    "x": lm.x * w,
                    "y": lm.y * h,
                    "z": lm.z * w,     # scale Z relative to width for perspective
                    "vis": lm.visibility
                })
        else:
            self.landmarks = None
        return self.landmarks

    def get_keypoints(self):
        """Returns dict of important joints for UI display."""
        if not self.landmarks:
            return None
        L = self.landmarks
        return {
            "HEAD": L[0],          # nose
            "LEFT_HAND": L[15],    # left wrist
            "RIGHT_HAND": L[16],   # right wrist
            "LEFT_FOOT": L[27],    # left ankle
            "RIGHT_FOOT": L[28],   # right ankle
        }