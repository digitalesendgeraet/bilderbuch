import sys
import os
import json
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtGui import QPixmap, QKeyEvent
from PyQt6.QtCore import Qt

# Tool um Goals einfacher zu setzen für die ganzen Bilder, Zeigt Bilder Automatisch an
# Wenn man 1 klick -> Goal 1 (reflektierende Kugel) + nächstes Bild; 0 -> Goal 0 (keine reflektierende Kugel) + nächstes Bild; u -> Goal -1 (unbracuhbares Bild)
# Mit Pfeiltasten kann man zurückgehen zu vorherigen Bildern und dann indem man entsprechende Taste Drückt neus Goal setzten; Bilder danach muss man Goals wieder neu setzen
# In der Json Datei kann man manuell eine 2 schreiben an der Stelle wo man Anfangen möchte zu Labeln, falls man schon etwas gemacht hat, ohen 2 beginn vorne, wenn man aufhöhrt automatisch 2 beim Letzten Bild gesetzt

JSON_PATH = "goals.json"
IMAGE_FOLDER = "formated_images"


class LabelTool(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(200, 200, 900, 900)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setGeometry(0, 0, 900, 900)

        self.load_json()

        self.pictures = self.data["pictures"]
        self.keys = list(self.pictures.keys())
        self.total = len(self.keys)

        self.current_index = self.find_resume_index()

        self.show_image()

    # ------------------------
    # JSON Handling
    # ------------------------

    def load_json(self):
        with open(JSON_PATH, "r") as f:
            self.data = json.load(f)

    def save_json(self):
        with open(JSON_PATH, "w") as f:
            json.dump(self.data, f, indent=4)

    # ------------------------
    # Resume Logic
    # ------------------------

    def find_resume_index(self):
        for i, key in enumerate(self.keys):
            if self.pictures[key]["goal"] == 2:
                return i
        return 0

    def clear_old_marker(self):
        for key in self.keys:
            if self.pictures[key]["goal"] == 2:
                self.pictures[key]["goal"] = 0  # temporary neutral reset

    def set_next_marker(self):
        if self.current_index + 1 < self.total:
            next_key = self.keys[self.current_index + 1]
            self.pictures[next_key]["goal"] = 2

    # ------------------------
    # UI Updates
    # ------------------------

    # def update_title(self):
    #     self.setWindowTitle(f"Image Label Tool  |  {self.current_index + 1} / {self.total}")

    def update_title(self):
        key = self.keys[self.current_index]
        self.setWindowTitle(f"{key}  |  {self.current_index}/{self.total - 1}")

    def show_image(self):
        if self.current_index < 0:
            self.current_index = 0

        if self.current_index >= self.total:
            print("Finished all images.")
            self.close()
            return

        key = self.keys[self.current_index]
        image_path = os.path.join(IMAGE_FOLDER, key)

        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            self.current_index += 1
            self.show_image()
            return

        pixmap = QPixmap(image_path)
        self.label.setPixmap(
            pixmap.scaled(
                self.label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        self.update_title()

        print(
            f"Showing: {key} | Goal: {self.pictures[key]['goal']} "
            f"| {self.current_index + 1}/{self.total}"
        )

    # ------------------------
    # Key Controls
    # ------------------------

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_0:
            self.set_goal(0)
        elif event.key() == Qt.Key.Key_1:
            self.set_goal(1)
        elif event.key() == Qt.Key.Key_U:
            self.set_goal(-1)
        elif event.key() == Qt.Key.Key_Left:
            self.go_back()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()

    # ------------------------
    # Core Label Logic
    # ------------------------

    def set_goal(self, value):
        key = self.keys[self.current_index]

        # Remove old marker
        self.clear_old_marker()

        # Set current label
        self.pictures[key]["goal"] = value

        # Set resume marker on next image
        self.set_next_marker()

        self.save_json()

        print(f"{key} set to {value}")

        self.current_index += 1
        self.show_image()

    def go_back(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelTool()
    window.show()
    sys.exit(app.exec())
