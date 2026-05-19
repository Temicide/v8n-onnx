import cv2
import numpy as np
import onnxruntime as ort
import threading

# ── Config ────────────────────────────────────────────────
MODEL_PATH   = "yolov8n.onnx"
INPUT_SIZE   = 640        # YOLOv8 default
CONF_THRESH  = 0.25

CLASSES = [               # COCO class names (80 classes)
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
# ─────────────────────────────────────────────────────────

def load_model(path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(path, providers=providers)
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] Model loaded — Input: {input_name}, Output: {output_name}")
    return session, input_name, output_name

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   
    img = np.expand_dims(img, axis=0)    
    return img

def postprocess_with_nms(outputs, orig_h, orig_w):
    detections = outputs[0]  
    if detections.ndim == 3:
        detections = detections[0]  

    boxes, scores, class_ids = [], [], []
    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < CONF_THRESH:
            continue

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))
        class_ids.append(int(cls_id))

    return boxes, scores, class_ids

def draw(frame, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# ── Camera Streams ────────────────────────────────────────
STREAMS = [
    "http://user7:rangsit1025@118.174.138.142:1025/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
    "http://user7:rangsit1033@118.174.138.142:1033/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
    "http://user7:rangsit1030@118.174.138.142:1030/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
    "http://user7:rangsit1031@118.174.138.142:1031/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
    "http://user7:rangsit1035@118.174.138.142:1035/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
    "http://user7:rangsit1029@118.174.138.142:1029/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
]
STREAM_LABELS = ["Cam1", "Cam2", "Cam3", "Cam4", "Cam5", "Cam6"]
GRID_COLS = 3   # 3x2 grid layout
GRID_W    = 640
GRID_H    = 360
# ─────────────────────────────────────────────────────────


class CameraStream:
    """Threaded stream reader so slow cameras don't block each other."""
    def __init__(self, url, label):
        self.url   = url
        self.label = label
        self.frame = None
        self.running = True
        self.cap   = cv2.VideoCapture(url)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                # Try to reconnect if stream drops
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()


def run_multi():
    session, input_name, output_name = load_model(MODEL_PATH)

    # Start all streams in background threads
    streams = [CameraStream(url, label)
               for url, label in zip(STREAMS, STREAM_LABELS)]

    print("[INFO] Connecting to streams... Press 'q' to quit.")

    while True:
        frames = []

        for stream in streams:
            frame = stream.get_frame()

            if frame is None:
                # Show black placeholder if stream not ready yet
                placeholder = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"{stream.label}: Connecting...",
                            (10, GRID_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (100, 100, 100), 2)
                frames.append(placeholder)
                continue

            # Run detection
            orig_h, orig_w = frame.shape[:2]
            inp = preprocess(frame)
            outputs = session.run([output_name], {input_name: inp})
            boxes, scores, class_ids = postprocess_with_nms(outputs, orig_h, orig_w)
            frame = draw(frame, boxes, scores, class_ids)

            # Label which camera
            cv2.putText(frame, stream.label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Resize to grid cell size
            frame = cv2.resize(frame, (GRID_W, GRID_H))
            frames.append(frame)

        # Arrange into 3x2 grid
        rows = []
        for i in range(0, len(frames), GRID_COLS):
            row_frames = frames[i:i + GRID_COLS]
            # Pad row if less than GRID_COLS cameras
            while len(row_frames) < GRID_COLS:
                row_frames.append(np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8))
            rows.append(np.hstack(row_frames))

        grid = np.vstack(rows)
        cv2.imshow("YOLOv8n — CCTV Detection", grid)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for stream in streams:
        stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_multi()