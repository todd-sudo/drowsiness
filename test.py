import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("video_2023-05-20_11-37-39.mp4")

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # detect faces in the image
        results = face_detection.process(image)

        if results.detections:
            for detection in results.detections:
                # extract the eye landmarks
                left_eye = detection.location_data.relative_keypoints[mp_face_detection.FaceKeyPoint.LEFT_EYE]
                right_eye = detection.location_data.relative_keypoints[mp_face_detection.FaceKeyPoint.RIGHT_EYE]

                # calculate the eye aspect ratio (EAR) to determine drowsiness
                ear = (left_eye.y - left_eye.x + right_eye.y - right_eye.x) / (2 * (left_eye.x - right_eye.x))

                # do something with the EAR value

        # display the image with the face and eye detections
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Driver Drowsiness Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()