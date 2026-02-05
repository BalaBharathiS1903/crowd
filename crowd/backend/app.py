from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS
import cv2
import datetime
import csv
import threading
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
CORS(app)

# 🔥 Use slightly stronger model than nano
model = YOLO("yolov8s.pt")  # more accurate than yolov8n

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

current_frame = None
current_count = 0
current_status = "SAFE"
people_data = []
count_history = []  # for smoothing


# 🎯 Detection Thread
def detect_people():
    global current_frame, current_count, current_status

    while True:
        success, frame = camera.read()
        if not success:
            continue

        h, w, _ = frame.shape
        frame_area = h * w

        # Track = removes duplicate detections across frames
        results = model.track(
            frame,
            persist=True,
            classes=[0],      # 0 = person only
            conf=0.65,        # higher confidence = fewer fake detections
            iou=0.5,
            verbose=False
        )

        count = 0

        if results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / frame_area

                # 🚫 Remove tiny false detections (like posters / shadows)
                if area_ratio < 0.015:  # must be at least 1.5% of frame
                    continue

                count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 📉 Smooth count to remove sudden spikes
        count_history.append(count)
        if len(count_history) > 5:
            count_history.pop(0)

        smooth_count = int(np.mean(count_history))

        current_count = smooth_count
        current_status = "OVERLOAD" if smooth_count > 3 else "SAFE"
        current_frame = frame

        people_data.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "count": smooth_count
        })

        if len(people_data) > 100:
            people_data.pop(0)


threading.Thread(target=detect_people, daemon=True).start()


# 🎥 Video Stream
def generate_frames():
    global current_frame
    while True:
        if current_frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', current_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: /jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/count')
def get_count():
    return jsonify({"count": current_count, "status": current_status})


@app.route('/data')
def get_data():
    return jsonify(people_data[-30:])


@app.route('/download')
def download_csv():
    filename = "crowd_data.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Count"])
        for row in people_data:
            writer.writerow([row["time"], row["count"]])
    return send_file(filename, as_attachment=True)


@app.route('/capture')
def capture_image():
    if current_frame is not None:
        filename = "capture.jpg"
        cv2.imwrite(filename, current_frame)
        return send_file(filename, as_attachment=True)
    return "No frame", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
