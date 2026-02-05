import cv2

# Pre-trained people detector (HOG + SVM)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def count_people():
    cap = cv2.VideoCapture(0)  # 0 = webcam

    if not cap.isOpened():
        return 0

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return 0

    frame = cv2.resize(frame, (640, 480))

    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))

    return len(boxes)
