import cv2
import numpy as np
import mediapipe as mp
from math import acos, degrees

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def finger_extended(lm, tip, pip):
    """Check if a finger is extended by comparing tip and pip y-coordinates"""
    return lm[tip].y < lm[pip].y


def classify_gesture(lm, handedness):
    """
    Classify hand gestures based on finger positions
    Returns gesture name as string
    """
    # Finger tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
    # Finger pip indices: thumb=2, index=6, middle=10, ring=14, pinky=18

    # Check thumb (use x-axis for thumb since it extends sideways)
    if handedness == "Right":
        thumb = lm[4].x > lm[3].x
    else:
        thumb = lm[4].x < lm[3].x

    # Check other fingers (use y-axis)
    index = finger_extended(lm, 8, 6)
    middle = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pinky = finger_extended(lm, 20, 18)

    extended_count = sum([thumb, index, middle, ring, pinky])

    # Classify gestures
    if extended_count == 5:
        return "OPEN HAND"
    elif extended_count == 0:
        return "FIST"
    elif index and middle and not ring and not pinky and not thumb:
        return "PEACE/VICTORY"
    elif index and not middle and not ring and not pinky and not thumb:
        return "POINTING"
    elif thumb and not index and not middle and not ring and not pinky:
        if lm[4].y < lm[0].y:  # thumb pointing up
            return "THUMBS UP"
        return "THUMB OUT"
    elif thumb and index and not middle and not ring and not pinky:
        return "GUN/L-SHAPE"
    elif pinky and not ring and not middle and not index and not thumb:
        return "PINKY UP"
    elif extended_count == 3 and middle and ring and pinky:
        return "THREE FINGERS"
    elif extended_count == 4 and not thumb:
        return "FOUR FINGERS"
    else:
        return f"{extended_count} FINGERS"


def get_bounding_box(lm, w, h, padding=30):
    """Calculate bounding box around hand landmarks with padding"""
    x_coords = [landmark.x * w for landmark in lm]
    y_coords = [landmark.y * h for landmark in lm]

    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(h, int(max(y_coords)) + padding)

    return (x_min, y_min, x_max, y_max)


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check permissions or try VideoCapture(1)")

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
    ) as hands:

        print("Hand Gesture Detection Started!")
        print("Press 'Q' to quit")

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam")
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Process detected hands
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness_info in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                ):
                    # Get handedness (Left/Right)
                    handedness = handedness_info.classification[0].label
                    confidence = handedness_info.classification[0].score

                    # Get landmarks
                    lm = hand_landmarks.landmark

                    # Classify gesture
                    gesture = classify_gesture(lm, handedness)

                    # Get bounding box
                    x_min, y_min, x_max, y_max = get_bounding_box(lm, w, h)

                    # Draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Draw hand landmarks and connections
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )

                    # Prepare label text
                    label = f"{handedness} Hand"
                    gesture_text = f"Gesture: {gesture}"

                    # Draw label background
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    gesture_size, _ = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                    # Draw filled rectangles for text background
                    cv2.rectangle(frame,
                                  (x_min, y_min - 50),
                                  (x_min + max(label_size[0], gesture_size[0]) + 10, y_min),
                                  (0, 255, 0), -1)

                    # Draw text labels
                    cv2.putText(frame, label, (x_min + 5, y_min - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(frame, gesture_text, (x_min + 5, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw instructions
            cv2.putText(frame, "Press 'Q' to quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Show frame count (optional)
            cv2.putText(frame, "Hand Gesture Detection", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Display output
            cv2.imshow("Hand Gesture Detection", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:  # Q or ESC
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended")


if __name__ == "__main__":
    main()