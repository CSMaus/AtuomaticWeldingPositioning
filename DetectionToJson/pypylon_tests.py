import sys
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from pypylon import pylon
import numpy as np
from time import time


class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.camera.PixelFormat.SetValue("BayerBG8")
        self.camera.MaxNumBuffer = 3
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.running = True

    def run(self):
        while self.running:
            st = time()
            grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab_result and grab_result.GrabSucceeded():
                img = grab_result.Array.copy()
                grab_result.Release()
                frame = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
                h, w, _ = frame.shape
                qimg = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(qimg)
            ed  = time()
            print("Image grab took:", ed - st)

    def stop(self):
        self.running = False
        self.camera.StopGrabbing()
        self.camera.Close()
        self.wait()


class CameraGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("No-Freeze Basler GUI")
        self.layout = QVBoxLayout(self)
        self.video_label = QLabel("Camera feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.btn = QPushButton("Start")
        self.layout.addWidget(self.btn)
        self.btn.clicked.connect(self.toggle_camera)

        self.cam_thread = None

    def toggle_camera(self):
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
            self.cam_thread = None
            self.btn.setText("Start")
        else:
            self.cam_thread = CameraThread()
            self.cam_thread.frame_ready.connect(self.update_image)
            self.cam_thread.start()
            self.btn.setText("Stop")

    def update_image(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        if self.cam_thread:
            self.cam_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CameraGUI()
    gui.resize(800, 600)
    gui.show()
    sys.exit(app.exec())
