import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

class EmotionDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Detection")

        self.start_button = QPushButton("Start Emotion Detection")
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)

        self.label = QLabel()
        self.label.setFixedSize(640, 480)

        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.emotion_detection_running = False
        self.facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def start_detection(self):
        if not self.emotion_detection_running:
            self.emotion_detection_running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            # Create the model
            model = Sequential()

            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(7, activation='softmax'))

            # Load pre-trained model weights
            model.load_weights('model.h5')

            # prevents openCL usage and unnecessary logging messages
            cv2.ocl.setUseOpenCL(False)

            # dictionary which assigns each label an emotion (alphabetical order)
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

            # start the webcam feed
            cap = cv2.VideoCapture(0)
            while self.emotion_detection_running:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Convert image to RGB format
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                # Convert image to QImage
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # Set image to label
                self.label.setPixmap(QPixmap(q_img))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        else:
            QMessageBox.information(self, "Emotion Detection", "Emotion detection is already running.")

    def stop_detection(self):
        self.emotion_detection_running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

if __name__ == "__main__":
    app = QApplication([])
    window = EmotionDetectionApp()
    window.show()
    app.exec_()
