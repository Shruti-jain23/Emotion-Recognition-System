import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('ck_emotion_recognition_cnn.h5')

# Emotion labels based on CK+ dataset (modify if different)
emotion_labels = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']

# Function to preprocess the face for prediction
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face = cv2.resize(face, (48, 48))  # Resize to the input size of the model
    face = face / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=-1)  # Add channel dimension
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using OpenCV's Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face
        face = frame[y:y+h, x:x+w]

        # Preprocess the face
        preprocessed_face = preprocess_face(face)

        # Predict emotion
        prediction = model.predict(preprocessed_face)
        emotion = emotion_labels[np.argmax(prediction)]

        # Display the emotion on the frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
