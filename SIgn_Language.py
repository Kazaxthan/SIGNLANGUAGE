import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

while True:
    # Read frame from the camera
    ret, image = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    result = hands.process(image)

    # Convert the image back to BGR for rendering
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw landmarks if hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mphands.HAND_CONNECTIONS)

    # Show the frame with landmarks
    cv2.imshow("Handtracker", image)

    # Check for key events
    key = cv2.waitKey(1) & 0xFF

    # If the 'x' key is pressed, break the loop to stop the camera
    if key == ord('x'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
