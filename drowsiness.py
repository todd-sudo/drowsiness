import random
import time
from typing import Any
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import \
    _normalized_to_pixel_coordinates as denormalize_coordinates


def distance(point_1, point_2):
    """Вычислить l2-норму между двумя точками"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(
        landmarks,
        refer_idxs: list,
        frame_width: int,
        frame_height: int
) -> tuple[float, list[tuple[int, int] | None]]:
    """ Получение глаза, вычислением евклидово расстояние между горизонталями
    """
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(
                lm.x, lm.y, frame_width, frame_height
            )
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except Exception as err:
        print(err)
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(
        landmarks,
        left_eye_idxs,
        right_eye_idxs,
        image_w,
        image_h
) -> tuple[
    float | Any,
    tuple[
        list[tuple[int, int] | None] | None,
        list[tuple[int, int] | None] | None
    ]
]:
    """ Рассчитывает соотношение сторон глаза
    """

    left_ear, left_lm_coordinates = get_ear(
        landmarks,
        left_eye_idxs,
        image_w,
        image_h
    )
    right_ear, right_lm_coordinates = get_ear(
        landmarks,
        right_eye_idxs,
        image_w,
        image_h
    )
    avg_ear = (left_ear + right_ear) / 2.0

    return avg_ear, (left_lm_coordinates, right_lm_coordinates)


def get_mediapipe_face_mesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
) -> mp.solutions.face_mesh.FaceMesh:
    """ Инициализирует и возвращает объект Mediapipe FaceMesh Solution Graph
    """
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh


def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    frame = cv2.flip(frame, 1)
    return frame


def plot_text(image, text, origin,
              color, font=cv2.FONT_HERSHEY_SIMPLEX,
              fntScale=0.8, thickness=2
              ):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


class VideoFrameHandler:
    def __init__(self):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        # Left and right eye chosen landmarks.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR

        self.facemesh_model = get_mediapipe_face_mesh()

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds time passed with EAR < EAR_THRESH
            "COLOR": self.GREEN,
            "play_alarm": False,
        }

        self.EAR_txt_pos = (10, 30)

    def process(self, frame: np.array, thresholds: dict):
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape
        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks,
                                                 self.eye_idxs["left"],
                                                 self.eye_idxs["right"],
                                                 frame_w,
                                                 frame_h
                                                 )
            frame = plot_eye_landmarks(frame,
                                       coordinates[0],
                                       coordinates[1],
                                       self.state_tracker["COLOR"]
                                       )

            if EAR < thresholds["EAR_THRESH"]:

                # Increase DROWSY_TIME to track the time period with
                # EAR less than the threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()

                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, "WAKE UP! WAKE UP",
                              ALM_txt_pos, self.state_tracker["COLOR"])

            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

            EAR_txt = f"EAR: {round(EAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
            plot_text(frame, EAR_txt,
                      self.EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, DROWSY_TIME_txt,
                      DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])

        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_alarm"]


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
