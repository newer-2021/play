import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection  # 人脸检测器
mp_drawing = mp.solutions.drawing_utils  # 绘制人脸地标

cap = cv2.VideoCapture('video/head3.MP4')  # 视频输入源

with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将图片转化为标准RGB格式

        people = 0
        if results.detections:
            for detection in results.detections:  # 获取人脸地标
                people += 1
                mp_drawing.draw_detection(image, detection)

        print(fr'[视频中人数：{people}] ')

        # 设置窗口大小
        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", 500, 460)
        cv2.imshow('image', image)

        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
