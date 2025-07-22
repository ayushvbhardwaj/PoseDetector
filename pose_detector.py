import cv2
import mediapipe as mp
import math

class GestureRecognizer:
    """
    A class to handle real-time gesture recognition using OpenCV and MediaPipe.

    This class encapsulates video capture, hand and pose landmark detection,
    and gesture interpretation with temporal smoothing to prevent flickering.
    This version includes optimized logic for more reliable gesture detection.
    """

    def __init__(self, consecutive_frames_threshold=10):
        """
        Initializes the GestureRecognizer.

        Args:
            consecutive_frames_threshold (int): The number of consecutive frames a gesture
                                                must be detected before it's confirmed.
        """
        # --- Constants and Configuration ---
        self.CONSECUTIVE_FRAMES_THRESHOLD = consecutive_frames_threshold

        # --- MediaPipe Initialization ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # --- Video Capture Initialization ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open camera.")
            exit()

        # --- State Management for Temporal Smoothing ---
        self.active_gesture = "Hover"
        self.potential_gesture = "Hover"
        self.gesture_counter = 0

    def _get_landmark_coords(self, landmarks, landmark_name):
        """
        Retrieves the x, y coordinates of a specific landmark.

        Args:
            landmarks: The list of detected landmarks from MediaPipe.
            landmark_name (str): The name of the landmark constant from mp_pose.PoseLandmark.

        Returns:
            tuple: A tuple (x, y) of the landmark's coordinates, or (None, None) if not found.
        """
        try:
            landmark_enum = self.mp_pose.PoseLandmark[landmark_name]
            landmark = landmarks.landmark[landmark_enum]
            return landmark.x, landmark.y
        except (KeyError, IndexError):
            return None, None

    # --- Optimized Gesture Definition Methods ---

    def is_takeoff(self, landmarks):
        """Checks for the 'Takeoff' gesture: both hands raised high and apart."""
        left_wrist_y = self._get_landmark_coords(landmarks, 'LEFT_WRIST')[1]
        right_wrist_y = self._get_landmark_coords(landmarks, 'RIGHT_WRIST')[1]
        left_shoulder_y = self._get_landmark_coords(landmarks, 'LEFT_SHOULDER')[1]
        right_shoulder_y = self._get_landmark_coords(landmarks, 'RIGHT_SHOULDER')[1]
        left_elbow_y = self._get_landmark_coords(landmarks, 'LEFT_ELBOW')[1]
        right_elbow_y = self._get_landmark_coords(landmarks, 'RIGHT_ELBOW')[1]

        if all(coord is not None for coord in [left_wrist_y, right_wrist_y, left_shoulder_y, right_shoulder_y, left_elbow_y, right_elbow_y]):
            # Check if both wrists are above their respective elbows, and elbows are above shoulders.
            return (left_wrist_y < left_elbow_y and left_elbow_y < left_shoulder_y and
                    right_wrist_y < right_elbow_y and right_elbow_y < right_shoulder_y)
        return False

    def is_move_right(self, landmarks):
        """
        Checks for 'Move Right': right arm extended outward (wrist->elbow->shoulder),
        left arm relaxed.
        """
        # Get all required coordinates
        right_wrist_x, right_wrist_y = self._get_landmark_coords(landmarks, 'RIGHT_WRIST')
        right_elbow_x, _ = self._get_landmark_coords(landmarks, 'RIGHT_ELBOW')
        right_shoulder_x, right_shoulder_y = self._get_landmark_coords(landmarks, 'RIGHT_SHOULDER')
        left_wrist_y = self._get_landmark_coords(landmarks, 'LEFT_WRIST')[1]
        left_shoulder_y = self._get_landmark_coords(landmarks, 'LEFT_SHOULDER')[1]

        if all(coord is not None for coord in [right_wrist_x, right_wrist_y, right_elbow_x, right_shoulder_x,
                                             left_wrist_y, left_shoulder_y]):
            # Check for proper arm extension sequence
            right_arm_extended = (right_wrist_x < right_elbow_x < right_shoulder_x)
            # Check if right arm is roughly at shoulder height
            right_arm_height_ok = abs(right_wrist_y - right_shoulder_y) < 0.15
            # More flexible check for left arm being relaxed
            left_arm_down = left_wrist_y > left_shoulder_y + 0.1
            
            return right_arm_extended and right_arm_height_ok and left_arm_down
        return False

    def is_move_left(self, landmarks):
        """
        Checks for 'Move Left': left arm extended outward (wrist->elbow->shoulder),
        right arm relaxed.
        """
        # Get all required coordinates
        left_wrist_x, left_wrist_y = self._get_landmark_coords(landmarks, 'LEFT_WRIST')
        left_elbow_x, _ = self._get_landmark_coords(landmarks, 'LEFT_ELBOW')
        left_shoulder_x, left_shoulder_y = self._get_landmark_coords(landmarks, 'LEFT_SHOULDER')
        right_wrist_y = self._get_landmark_coords(landmarks, 'RIGHT_WRIST')[1]
        right_shoulder_y = self._get_landmark_coords(landmarks, 'RIGHT_SHOULDER')[1]

        if all(coord is not None for coord in [left_wrist_x, left_wrist_y, left_elbow_x, left_shoulder_x,
                                             right_wrist_y, right_shoulder_y]):
            # Check for proper arm extension sequence
            left_arm_extended = (left_wrist_x > left_elbow_x > left_shoulder_x)
            # Check if left arm is roughly at shoulder height
            left_arm_height_ok = abs(left_wrist_y - left_shoulder_y) < 0.15
            # More flexible check for right arm being relaxed
            right_arm_down = right_wrist_y > right_shoulder_y + 0.1
            
            return left_arm_extended and left_arm_height_ok and right_arm_down
        return False

    def is_land(self, landmarks):
        """Checks for the 'Land' gesture: arms crossed over chest."""
        left_wrist_x, left_wrist_y = self._get_landmark_coords(landmarks, 'LEFT_WRIST')
        right_wrist_x, right_wrist_y = self._get_landmark_coords(landmarks, 'RIGHT_WRIST')
        left_shoulder_x, left_shoulder_y = self._get_landmark_coords(landmarks, 'LEFT_SHOULDER')
        right_shoulder_x, right_shoulder_y = self._get_landmark_coords(landmarks, 'RIGHT_SHOULDER')

        if all(coord is not None for coord in [left_wrist_x, right_wrist_x, 
                                            left_shoulder_x, right_shoulder_x,
                                            left_wrist_y, right_wrist_y, 
                                            left_shoulder_y, right_shoulder_y]):
            # Check if arms are crossed (right hand on left side and left hand on right side)
            is_crossed = (right_wrist_x < left_shoulder_x and 
                        left_wrist_x > right_shoulder_x)
            
            # Check if hands are at chest height
            shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
            hands_at_chest = (abs(left_wrist_y - shoulder_y) < 0.2 and 
                            abs(right_wrist_y - shoulder_y) < 0.2)
            
            return is_crossed and hands_at_chest
        return False

    def is_hover(self, landmarks):
        """Checks for 'Hover': a neutral pose with both hands down by the sides."""
        left_wrist_y = self._get_landmark_coords(landmarks, 'LEFT_WRIST')[1]
        right_wrist_y = self._get_landmark_coords(landmarks, 'RIGHT_WRIST')[1]
        left_hip_y = self._get_landmark_coords(landmarks, 'LEFT_HIP')[1]
        right_hip_y = self._get_landmark_coords(landmarks, 'RIGHT_HIP')[1]
        
        if all(coord is not None for coord in [left_wrist_y, right_wrist_y, 
                                             left_hip_y, right_hip_y]):
            # More lenient checks for arms being down
            left_arm_down = left_wrist_y > left_hip_y - 0.1
            right_arm_down = right_wrist_y > right_hip_y - 0.1
            return left_arm_down and right_arm_down
        return False

    def is_move_forward(self, landmarks):
        """
        Checks for 'Move Forward': both arms extended horizontally towards the camera.
        Arms should be at shoulder height and roughly equal distance from shoulders.
        """
        # Get coordinates for both arms
        left_wrist_x, left_wrist_y = self._get_landmark_coords(landmarks, 'LEFT_WRIST')
        right_wrist_x, right_wrist_y = self._get_landmark_coords(landmarks, 'RIGHT_WRIST')
        left_shoulder_x, left_shoulder_y = self._get_landmark_coords(landmarks, 'LEFT_SHOULDER')
        right_shoulder_x, right_shoulder_y = self._get_landmark_coords(landmarks, 'RIGHT_SHOULDER')
        left_elbow_x, _ = self._get_landmark_coords(landmarks, 'LEFT_ELBOW')
        right_elbow_x, _ = self._get_landmark_coords(landmarks, 'RIGHT_ELBOW')

        if all(coord is not None for coord in [
            left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y,
            left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y,
            left_elbow_x, right_elbow_x
        ]):
            # Calculate shoulder midpoint x-coordinate
            shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2

            # Check if both arms are at shoulder height (more lenient)
            left_arm_height_ok = abs(left_wrist_y - left_shoulder_y) < 0.2
            right_arm_height_ok = abs(right_wrist_y - right_shoulder_y) < 0.2

            # Check if arms are roughly centered (wrists should be close to shoulder center)
            left_arm_centered = abs(left_wrist_x - shoulder_center_x) < 0.2
            right_arm_centered = abs(right_wrist_x - shoulder_center_x) < 0.2

            # Check if elbows are between wrists and shoulders
            left_elbow_ok = min(left_wrist_x, left_shoulder_x) < left_elbow_x < max(left_wrist_x, left_shoulder_x)
            right_elbow_ok = min(right_wrist_x, right_shoulder_x) < right_elbow_x < max(right_wrist_x, right_shoulder_x)

            # All conditions must be met
            return all([
                left_arm_height_ok, right_arm_height_ok,
                left_arm_centered, right_arm_centered,
                left_elbow_ok, right_elbow_ok
            ])
        return False

    def _update_gesture_state(self, new_gesture):
        """
        Manages the state of gestures for temporal smoothing.
        Updates counters and confirms gestures if they are held consistently.
        """
        if new_gesture == self.potential_gesture:
            self.gesture_counter += 1
            if self.gesture_counter >= self.CONSECUTIVE_FRAMES_THRESHOLD:
                # Avoid resetting active gesture if it's already set to the potential one
                if self.active_gesture != self.potential_gesture:
                    self.active_gesture = self.potential_gesture
        else:
            self.potential_gesture = new_gesture
            self.gesture_counter = 1 # Reset counter for the new potential gesture

    def _draw_feedback_on_image(self, image):
        """
        Draws visual feedback on the image, including active and potential gestures.
        """
        # --- Display Text Configuration ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        active_color = (0, 255, 0)  # Green
        potential_color = (0, 255, 255) # Yellow
        thickness = 2
        
        # --- Display Active Gesture ---
        active_text = f"Active Gesture: {self.active_gesture}"
        cv2.putText(image, active_text, (50, 50), font, 1, active_color, thickness, cv2.LINE_AA)

        # --- Display Potential Gesture and Counter ---
        if self.potential_gesture != self.active_gesture and self.potential_gesture != "Unknown":
            potential_text = f"Detecting {self.potential_gesture}... ({self.gesture_counter}/{self.CONSECUTIVE_FRAMES_THRESHOLD})"
            cv2.putText(image, potential_text, (50, 100), font, 0.8, potential_color, thickness, cv2.LINE_AA)

    def run(self):
        """
        The main loop for capturing video, processing frames, and detecting gestures.
        """
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # --- Gesture Detection Logic ---
            current_gesture = "Unknown"
            if results.pose_landmarks:
                # The order of checks is important. Check for more specific/demanding
                # gestures first.
                if self.is_takeoff(results.pose_landmarks):
                    current_gesture = "Takeoff"
                elif self.is_land(results.pose_landmarks):
                    current_gesture = "Land"
                elif self.is_move_right(results.pose_landmarks):
                    current_gesture = "Move Right"
                elif self.is_move_left(results.pose_landmarks):
                    current_gesture = "Move Left"
                elif self.is_move_forward(results.pose_landmarks):
                    current_gesture = "Move Forward"
                elif self.is_hover(results.pose_landmarks):
                    # Hover is the default/fallback pose, check it last.
                    current_gesture = "Hover"
                
                self._update_gesture_state(current_gesture)

                # --- Drawing Landmarks and Feedback ---
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            else:
                # If no landmarks are detected, reset to a neutral state
                self._update_gesture_state("Hover")

            self._draw_feedback_on_image(image)
            
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Gesture Recognition', cv2.flip(image, 1))

            if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' to exit
                break
        
        # --- Cleanup ---
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

if __name__ == '__main__':
    # To run the gesture recognizer, create an instance of the class and call the run method.
    recognizer = GestureRecognizer(consecutive_frames_threshold=5)  # Changed from 10 to 5
    recognizer.run()
