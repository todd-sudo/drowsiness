import json
import os.path
import time

import cv2
import numpy as np

from settings.base import AppSettings
from src.ear import EARHelperMixin
from src.message import MessageHelperMixin
from src.mp import MediaPipeHelper


COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0)
}


class DrowsinessDetector(EARHelperMixin, MessageHelperMixin):
    def __init__(self):

        if not os.path.exists("settings.json"):
            exit("Нет файла конфигурации settings.json")

        # get config
        with open("settings.json", "r") as file:
            data = json.load(file)
        self.cfg = AppSettings()
        self.cfg.dynamic_settings = self.cfg.dynamic_settings.from_dict(data)

        self.input_video = self.cfg.dynamic_settings.input_video
        if not self.input_video:
            exit("Ошибка загрузки видео. Проверьте путь к видеофайлу")

        self.eye_idxs = self.cfg.static_settings.eye_idxs

        refine_landmarks: bool = True

        self.face_mesh_helper = MediaPipeHelper(
            max_num_faces=self.cfg.dynamic_settings.max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=self.cfg.static_settings \
                .min_detection_confidence,
            min_tracking_confidence=self.cfg.static_settings \
                .min_tracking_confidence,
        )
        self.face_mesh = self.face_mesh_helper.get_mediapipe_face_mesh()

        self.state_tracker = {
            "start_time": time.perf_counter(),
            "drowsy_time": 0.0,
            "color": COLORS["GREEN"],
            "play_alarm": False,
        }

        self.ear_txt_pos = (10, 30)

    def _detect(
            self, frame: np.array, thresholds: dict
    ) -> tuple[np.array, bool]:
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        drowsy_time_txt_pos = (10, int(frame_h // 2 * 1.7))
        alm_txt_pos = (10, int(frame_h // 2 * 1.85))

        results = self.face_mesh.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear, coordinates = self.calculate_avg_ear(
                landmarks,
                self.eye_idxs["left"],
                self.eye_idxs["right"],
                frame_w,
                frame_h
            )
            frame = self.plot_eye_landmarks(
                frame,
                coordinates[0],
                coordinates[1],
                self.state_tracker["color"]
            )

            if ear < thresholds["ear_thresh"]:

                end_time = time.perf_counter()

                self.state_tracker["drowsy_time"] += \
                    end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["color"] = COLORS["RED"]

                if self.state_tracker["drowsy_time"] >= \
                        thresholds["wait_time"]:
                    self.state_tracker["play_alarm"] = True
                    self.plot_text(
                        frame,
                        "WAKE UP! WAKE UP",
                        alm_txt_pos, self.state_tracker["color"])

            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["drowsy_time"] = 0.0
                self.state_tracker["color"] = COLORS["GREEN"]
                self.state_tracker["play_alarm"] = False

            ear_txt = f"EAR: {round(ear, 2)}"
            drowsy_time_txt = f"DROWSY: {round(self.state_tracker['drowsy_time'], 3)} Secs"
            self.plot_text(
                frame,
                ear_txt,
                self.ear_txt_pos,
                self.state_tracker["color"]
            )
            self.plot_text(
                frame,
                drowsy_time_txt,
                drowsy_time_txt_pos,
                self.state_tracker["color"]
            )

        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["color"] = COLORS["GREEN"]
            self.state_tracker["play_alarm"] = False

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)
        return frame, self.state_tracker["play_alarm"]

    def run_detector(self):

        cap = cv2.VideoCapture(self.input_video)

        thresholds = {
            "ear_thresh": self.cfg.dynamic_settings.ear_thresh,
            "wait_time": self.cfg.dynamic_settings.wait_time,
        }

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.resize(frame, (400, 700))

            frame, _ = self._detect(frame=frame, thresholds=thresholds)

            cv2.imshow("drowsiness", frame)

            if cv2.waitKey(10) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    DrowsinessDetector().run_detector()


if __name__ == '__main__':
    main()
