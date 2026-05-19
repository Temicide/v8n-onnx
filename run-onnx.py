import cv2
import numpy as np
import onnxruntime as ort

# ── Config ────────────────────────────────────────────────
MODEL_PATH   = "yolov8n.onnx"
SOURCE       = 0          # 0 = webcam, or "video.mp4", or "image.jpg"
INPUT_SIZE   = 640        # YOLOv8 default
CONF_THRESH  = 0.25
IOU_THRESH   = 0.45       # used only if NMS is NOT baked in

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
    """Resize + normalize frame for YOLOv8 input."""
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, axis=0)    # add batch dim → (1, 3, 640, 640)
    return img


def postprocess_with_nms(outputs, orig_h, orig_w):
    """
    Parse YOLOv8 ONNX output (with NMS baked in).
    Output shape: (1, num_detections, 6) → [x1, y1, x2, y2, conf, class_id]
    """
    detections = outputs[0]  # shape: (1, N, 6)
    if detections.ndim == 3:
        detections = detections[0]  # → (N, 6)

    boxes, scores, class_ids = [], [], []

    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < CONF_THRESH:
            continue

        # Scale back to original image size
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


def run():
    session, input_name, output_name = load_model(MODEL_PATH)

    # Handle image / video / webcam
    if isinstance(SOURCE, str) and SOURCE.lower().endswith((".jpg", ".jpeg", ".png")):
        frame = cv2.imread(SOURCE)
        orig_h, orig_w = frame.shape[:2]
        inp = preprocess(frame)
        outputs = session.run([output_name], {input_name: inp})
        boxes, scores, class_ids = postprocess_with_nms(outputs, orig_h, orig_w)
        frame = draw(frame, boxes, scores, class_ids)
        cv2.imshow("YOLOv8n Detection", frame)
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(SOURCE)
        print("[INFO] Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig_h, orig_w = frame.shape[:2]
            inp = preprocess(frame)
            outputs = session.run([output_name], {input_name: inp})
            boxes, scores, class_ids = postprocess_with_nms(outputs, orig_h, orig_w)
            frame = draw(frame, boxes, scores, class_ids)

            cv2.imshow("YOLOv8n Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()