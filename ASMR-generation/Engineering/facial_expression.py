import threading
import time
import cv2
import mediapipe as mp


class FacialExpressionDetector:
    def __init__(self):
        # initialize mediapipe facial detection model
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # max number of faces
            refine_landmarks=True,  # detailed points
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # initialize drawing
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # define detection states
        self.is_running = False
        self.current_expression = "Face not detected"
        self.last_update_time = time.time()

        # # start thread
        # self.capture_thread = threading.Thread(target=self._capture_loop)
        # self.capture_thread.daemon = True

    def start_detection(self):
        if not self.is_running:
            self.is_running = True
            self._capture_loop()
            print("Start facial expression detection")

    def stop_detection(self):
        self.is_running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        print("Facial expression detection is stopped")

    def _capture_loop(self):
        cap = cv2.VideoCapture(0)

        while self.is_running and cap.isOpened():
            ret, image = cap.read()
            if not ret:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = self.face_mesh.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # analyze expression
            self._analyze_expression(results)

            # graph facial information on image
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # graph facial grid
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.drawing_spec
                    )
            # show current emotion on the top-left of the image
            cv2.putText(image, f"Expression: {self.current_expression}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Facial Expression Detection', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running=False
                break

        cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

    def _analyze_expression(self, results):
        if not results.multi_face_landmarks:
            self.current_expression = "Face not detected"
            return

        # Obtain landmarks on face
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark # Obtain landmarks

        # Detect smile
        left_mouth_corner = landmarks[61]
        right_mouth_corner = landmarks[291]
        upper_lip = landmarks[0] # center of upper lip
        lower_lip = landmarks[17] # center of lower lip
        # calculate mouth openess
        mouth_openness = abs(upper_lip.y - lower_lip.y)

        left_eyebrow = landmarks[105] # center of eyebrow
        right_eyebrow = landmarks[334]
        left_eye = landmarks[33] # left eye reference point
        right_eye = landmarks[263]


        # calculate distance between eyebrow and eye
        left_eyebrow_raise = abs(left_eye.y -left_eyebrow.y)
        right_eyebrow_raise = abs(right_eye.y - right_eyebrow.y)

        # calculate eye openness
        left_eye_top = landmarks[159] # left eye top
        left_eye_bottom = landmarks[145] # left eye bottom
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]

        left_eye_open = abs(left_eye_top.y- left_eye_bottom.y)
        right_eye_open = abs(right_eye_top.y - right_eye_bottom.y)

        # analysis logic
        expression = "Neutral"

        if mouth_openness > 0.05 and (upper_lip.y - 0.02) < left_mouth_corner.y < (upper_lip.y + 0.02):
            expression = "smiling"

        elif left_eye_open < 0.01 and right_eye_open < 0.01:
            expression = "eye_closed"

        # update current expression
        self.current_expression = expression

        # update every 0.5s
        current_time = time.time()
        if current_time - self.last_update_time > 0.5:
            print(f"Expression detected: {expression}")
            self.last_update_time = current_time

if __name__ == "__main__":
    detector = FacialExpressionDetector()
    detector.start_detection()
    time.sleep(30) # run for 30 seconds before stopping
    detector.stop_detection()









