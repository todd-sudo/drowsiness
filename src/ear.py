from typing import Any

from mediapipe.python.solutions.drawing_utils import \
    _normalized_to_pixel_coordinates as denormalize_coordinates


class EARHelperMixin:

    @staticmethod
    def _distance(point_1: tuple[int, int], point_2: tuple[int, int]) -> float:
        """ Вычислить l2-норму между двумя точками
        """
        dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
        return dist

    def _get_ear(
            self,
            landmarks,
            refer_idxs: list,
            frame_width: int,
            frame_height: int
    ) -> tuple[float, list[tuple[int, int] | None]]:
        """ Получение глаза, вычислением евклидово
        расстояние между горизонталями
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
            p2_p6 = self._distance(coords_points[1], coords_points[5])
            p3_p5 = self._distance(coords_points[2], coords_points[4])
            p1_p4 = self._distance(coords_points[0], coords_points[3])

            # Compute the eye aspect ratio
            ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)

        except Exception as err:
            print(err)
            ear = 0.0
            coords_points = None

        return ear, coords_points

    def calculate_avg_ear(
            self,
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

        left_ear, left_lm_coordinates = self._get_ear(
            landmarks,
            left_eye_idxs,
            image_w,
            image_h
        )
        right_ear, right_lm_coordinates = self._get_ear(
            landmarks,
            right_eye_idxs,
            image_w,
            image_h
        )
        avg_ear = (left_ear + right_ear) / 2.0

        return avg_ear, (left_lm_coordinates, right_lm_coordinates)



