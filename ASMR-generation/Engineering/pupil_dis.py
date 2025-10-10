import cv2
import mediapipe as mp
import math
import pandas as pd
import numpy as np

class PupilDetector:
    def __init__(self):
        self.data = []
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Define crucial point indices
        # Around the eye and pupil
        self.LEFT_EYE_INDICES = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        self.RIGHT_EYE_INDICES = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]

        self.LEFT_EAR_INDEX = 234
        self.RIGHT_EAR_INDEX = 454

        # pupil detection state
        self.ear_distance_px = 0 # on camera
        self.left_pupil_diameter_mm = 0
        self.right_pupil_diameter_mm = 0
        self.avg_pupil_diameter_mm = 0
        self.pixel_to_mm_factor = 0

        self.ear_distance_mm = 0

    def run(self):
        self._get_ear_distance()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera cannot be opened")
            return

        print("Pupil Detection started")
        print("Left pupil diameter (mm) \t Right pupil diameter (mm) \t Average diameter (mm)")


        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Cannot read image")
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = self.face_mesh.process(rgb_image)

            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            # analyze pupil and ear positions
            self._detect_pupils_and_ears(image, results)

            # analyze pupil

            cv2.imshow("Pupil Detector", image)

            key = cv2.waitKey(1)
            if key == 27:
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        if self.face_mesh:
            self.face_mesh.close()

        # At the end of run(), before print("Exit successful")
        if self.data:
            df = pd.DataFrame(self.data)
            df.to_csv("pupil_measurements.csv", index=False)
            print("Pupil data saved to pupil_measurements.csv")

        print("Exit successful")

    def _distance(self, p1, p2):
        # distance between 2 points
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _calculate_diameter(self, iris_landmarks, img_w, img_h):
        if len(iris_landmarks) < 4:
            return 0

        points = []
        for landmark in iris_landmarks:
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)
            points.append((x, y))

        left_point = min(points, key=lambda p: p[0])
        right_point = max(points, key=lambda p: p[0])
        horizontal_diameter = math.sqrt((right_point[0] - left_point[0]) ** 2 +
                                        (right_point[1] - left_point[1]) ** 2)

        top_point = min(points, key=lambda p: p[1])
        bottom_point = max(points, key=lambda p: p[1])
        vertical_diameter = math.sqrt((bottom_point[0] - top_point[0]) ** 2 +
                                        (bottom_point[1] - top_point[1]) ** 2)

        return (horizontal_diameter + vertical_diameter) / 2


    def _detect_ears(self, image, landmarks, img_w, img_h):
        # obtain and calculate distance between left and right ear

        left_ear = landmarks[self.LEFT_EAR_INDEX]
        right_ear = landmarks[self.RIGHT_EAR_INDEX]

        # Calculate distance
        self.ear_distance_px = self._distance(left_ear, right_ear) * img_w

        # calculate coordinate
        left_ear_x = int(left_ear.x * img_w)
        left_ear_y = int(left_ear.y * img_h)
        right_ear_x = int(right_ear.x * img_w)
        right_ear_y = int(right_ear.y * img_h)

        # show in image
        cv2.circle(image, (left_ear_x, left_ear_y), 5, (255, 0, 0), -1)
        cv2.circle(image, (right_ear_x, right_ear_y), 5, (255, 0, 0), -1)

        cv2.putText(image, f"Ear Distance: {self.ear_distance_px:.1f}px", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    def _detect_eye(self, image, iris_landmarks, img_w, img_h):
        # process single eye and pupil diameter
        if len(iris_landmarks) < 4:
            return 0

        # calculate pupil center
        center_x = int((sum(point.x for point in iris_landmarks)) / len(iris_landmarks) * img_w)
        center_y = int((sum(point.y for point in iris_landmarks)) / len(iris_landmarks) * img_h)

        # calculate pupil diameter
        diameter = self._calculate_diameter(iris_landmarks, img_w, img_h)

        # draw pupil center
        cv2.circle(image, (center_x, center_y), 3, (0, 255, 255), -1)

        # draw pupil sketch
        for landmark in iris_landmarks:
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)
            cv2.circle(image, (x, y), 3, (255, 255, 0), -1)

        return diameter

    def _calculate_actual_diameter(self, left_diameter_px, right_diameter_px):
        if self.ear_distance_mm > 0 and self.ear_distance_px > 0:
            self.pixel_to_mm_factor = self.ear_distance_mm / self.ear_distance_px

            # calculate actual diameter
            self.left_pupil_diameter_mm = left_diameter_px * self.pixel_to_mm_factor
            self.right_pupil_diameter_mm = right_diameter_px * self.pixel_to_mm_factor
            self.avg_pupil_diameter_mm = (self.left_pupil_diameter_mm + self.right_pupil_diameter_mm)/2

        else:
            self.left_pupil_diameter_mm = 0
            self.right_pupil_diameter_mm = 0
            self.avg_pupil_diameter_mm = 0




    def _detect_pupils_and_ears(self, image, results):
        # calculate positions and actual diameter
        img_h, img_w, _ = image.shape
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # detect ear positions
            self._detect_ears(image, landmarks, img_w, img_h)

            # detect pupil
            left_diameter_px = self._detect_eye(image, [landmarks[i] for i in self.LEFT_IRIS_INDICES], img_w, img_h)
            right_diameter_px = self._detect_eye(image, [landmarks[i] for i in self.RIGHT_IRIS_INDICES], img_w, img_h)

            # transfer into millimeters
            self._calculate_actual_diameter(left_diameter_px, right_diameter_px)

            self.data.append({
                "Left Pupil (mm)": self.left_pupil_diameter_mm,
                "Right Pupil (mm)": self.right_pupil_diameter_mm,
                "Average Pupil (mm)": self.avg_pupil_diameter_mm
            })

            # show in image
            cv2.putText(image, f"L: {self.left_pupil_diameter_mm: .1f}mm, R: {self.right_pupil_diameter_mm: .1f}mm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            print(f"{self.left_pupil_diameter_mm:.1f}\t\t{self.right_pupil_diameter_mm:.1f}\t\t{self.avg_pupil_diameter_mm:.1f}")
        else:
            # face not detected
            cv2.putText(image, "No Face Detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Reset data
            self.left_pupil_diameter_mm = 0
            self.right_pupil_diameter_mm = 0
            self.avg_pupil_diameter_mm = 0
            self.ear_distance_px = 0

    def _get_ear_distance(self):
        # User input ear distance (mm)
        while True:
            try:
                input_str = input("Please type in the distance between your left ear and right ear (mm): ")
                self.ear_distance_mm = float(input_str)
                if self.ear_distance_mm == 0:
                    print("Distance has to be greater than 0")
                    continue
                print(f"Ear distance: {self.ear_distance_mm}mm")
                break
            except ValueError:
                print("print valid number")

if __name__ == "__main__":
    detector = PupilDetector()
    try:
        detector.run()
    except KeyboardInterrupt:
        print("Program end")



