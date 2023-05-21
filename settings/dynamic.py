from dataclasses import dataclass


@dataclass
class DynamicSettings:
    ear_thresh: float
    wait_time: float
    input_video: str
    max_num_faces: int

    @staticmethod
    def from_dict(obj: dict) -> 'DynamicSettings':
        ear_thresh = float(obj.get("ear_thresh").get("value"))
        wait_time = float(obj.get("wait_time").get("value"))
        input_video = obj.get("input_video").get("value")
        max_num_faces = int(obj.get("max_num_faces").get("value"))
        return DynamicSettings(
            ear_thresh, wait_time, input_video, max_num_faces
        )
