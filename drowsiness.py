import random
import time
import cv2
import numpy as np

from src.ear import EARHelperMixin
from src.message import MessageHelperMixin
from src.mp import MediaPipeHelper


COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0)
}


class DrowsinessDetector(EARHelperMixin, MessageHelperMixin):
    def __init__(
            self,
            eye_idxs: dict,
            max_num_faces: int = 1,
            refine_landmarks: bool = True,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5,
    ):
        self.eye_idxs = eye_idxs

        self.face_mesh_helper = MediaPipeHelper(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.face_mesh = self.face_mesh_helper.get_mediapipe_face_mesh()

        self.state_tracker = {
            "start_time": time.perf_counter(),
            "drowsy_time": 0.0,
            "color": COLORS["GREEN"],
            "play_alarm": False,
        }

        self.ear_txt_pos = (10, 30)

    def _detect(self, frame: np.array, thresholds: dict):
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
                self.state_tracker["COLOR"]
            )

            if ear < thresholds["EAR_THRESH"]:

                end_time = time.perf_counter()

                self.state_tracker["drowsy_time"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["drowsy_time"] >= thresholds["wait_time"]:
                    self.state_tracker["play_alarm"] = True
                    self.plot_text(
                        frame,
                        "WAKE UP! WAKE UP",
                        alm_txt_pos, self.state_tracker["COLOR"])

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
            self.state_tracker["COLOR"] = COLORS["GREEN"]
            self.state_tracker["play_alarm"] = False

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_alarm"]


    def run(self):



def main():
    cap = cv2.VideoCapture("video_2023.mp4")
    i = VideoFrameHandler()
    thresholds = {
        "EAR_THRESH": random.choice([0.0, 0.4, 0.18, 0.01]),
        "WAIT_TIME": random.choice([0.0, 5.0, 1.0, 0.25]),
    }
    thresholds = {
        "EAR_THRESH": 0.15,
        "WAIT_TIME": 0.10,
    }
    # TODO: дописать запуск и конфигурацию всего приложения
    #  с разносом на динамическую настройку и статичную
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (400, 700))

        frame, s = i.process(frame, thresholds=thresholds)

        cv2.imshow("test", frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
