from flask import Flask, jsonify, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the emotion recognition model
model = load_model('ck_emotion_recognition_cnn.h5')
emotion_labels = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']

# Function to preprocess the face for prediction
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the frame data from the frontend
    data = request.get_json()
    image_data = data['image']
    
    # Decode the image
    image_data = image_data.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = np.array(image)
    
    # Convert image to RGB if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert image to grayscale and preprocess it for emotion prediction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotion = 'Neutral'  # Default emotion if no face detected

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        preprocessed_face = preprocess_face(face)
        prediction = model.predict(preprocessed_face)
        emotion = emotion_labels[np.argmax(prediction)]

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
