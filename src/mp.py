import mediapipe as mp


class MediaPipeHelper:

    def __init__(
            self,
            max_num_faces: int,
            refine_landmarks: bool,
            min_detection_confidence: float,
            min_tracking_confidence: float,
    ):
        self.max_num_faces: int = max_num_faces
        self.refine_landmarks: bool = refine_landmarks
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence

    def get_mediapipe_face_mesh(self) -> mp.solutions.face_mesh.FaceMesh:
        """ Инициализирует и возвращает объект
        Mediapipe FaceMesh Solution Graph
        """
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return face_mesh
