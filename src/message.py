import numpy as np
import cv2


class MessageHelperMixin:

    @staticmethod
    def plot_eye_landmarks(
            frame: np.array,
            left_lm_coordinates,
            right_lm_coordinates,
            color
    ):
        for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
            if lm_coordinates:
                for coord in lm_coordinates:
                    frame_copy = frame.copy()
                    cv2.circle(frame_copy, coord, 2, color, -1)

        frame = cv2.flip(frame, 1)
        return frame

    @staticmethod
    def plot_text(
            frame: np.array,
            text: str,
            origin,
            color,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            fnt_scale=0.8,
            thickness=2
    ):
        frame = cv2.putText(
            frame, text, origin, font, fnt_scale, color, thickness
        )
        return frame
