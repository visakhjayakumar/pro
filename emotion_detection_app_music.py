import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtCore import Qt, QTimer
from tensorflow.keras.models import load_model

class EmotionDetectionAppMusic(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotion Detection Music Player")
        self.setGeometry(100, 100, 800, 600)

        self.start_button = QPushButton("Start", self)
        self.start_button.setGeometry(100, 100, 100, 50)
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setGeometry(250, 100, 100, 50)
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)

        self.emotion_detection_running = False
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = load_model('model.h5')
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    def start_detection(self):
        if not self.emotion_detection_running:
            self.emotion_detection_running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = self.model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))
                    print(self.emotion_dict[maxindex])  # Replace this with music playback
                    cv2.putText(frame, self.emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Video', cv2.resize(frame,(800,600),interpolation = cv2.INTER_CUBIC))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    def stop_detection(self):
        self.emotion_detection_running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


if __name__ == '__main__':
    app = QApplication([])
    window = EmotionDetectionAppMusic()
    window.show()
    app.exec_()
