import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
from tensorflow.keras.models import load_model

class EmotionApp(QDialog):
    def __init__(self):
        super(EmotionApp, self).__init__()
        loadUi('emotion_app.ui', self)
        self.image = None
        self.model = load_model('model.h5')
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.captureButton.clicked.connect(self.capture_image)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.start(10)

    def stop_webcam(self):
        self.timer.stop()

    def capture_image(self):
        _, frame = self.capture.read()
        cv2.imwrite("captured_image.jpg", frame)
        self.image = frame
        self.display_image()

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image = cv2.resize(self.image, (640, 480))
            self.detect_and_display_emotion()
            self.display_image()

    def display_image(self):
        qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        self.videoLabel.setPixmap(QPixmap.fromImage(img))
        self.videoLabel.setScaledContents(True)

    def detect_and_display_emotion(self):
        if self.image is not None:
            emotion_label = self.detectEmotion(self.image)
            cv2.putText(self.image, emotion_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def detectEmotion(self, frame):
        # Preprocess frame for emotion detection (resize, normalize, etc.)
        resized_frame = cv2.resize(frame, (48, 48))
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)
        
        # Call your emotion detection model to predict emotion
        emotion_predictions = self.model.predict(input_frame)
        
        # Get the index of the predicted emotion
        predicted_emotion_index = np.argmax(emotion_predictions)
        
        # Map index to emotion label (assuming emotion_dict is defined)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        emotion_label = emotion_dict[predicted_emotion_index]
        
        # Return detected emotion
        return emotion_label

app = QApplication(sys.argv)
window = EmotionApp()
window.setWindowTitle('Emotion Detection App')
window.show()
sys.exit(app.exec_())
