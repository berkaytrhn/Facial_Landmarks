import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Setup face mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # Set to True for static images
    max_num_faces=1,  # Set the max number of faces to detect
    min_detection_confidence=0.5,  # Minimum confidence for detection
    min_tracking_confidence=0.5  # Minimum confidence for tracking
)

image = cv2.imread('face.jpg')

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image
results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

# Show the image with landmarks
cv2.imshow('Face Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()