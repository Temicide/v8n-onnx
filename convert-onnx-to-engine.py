# -*- coding: utf-8 -*-
"""YOLOv8n CCTV Detection + TensorRT Engine Conversion (Colab T4)"""

import os
import sys
import subprocess
import threading
import warnings

# ═══════════════════════════════════════════════════════════
# 1. Dependency Setup (safe for .py and Colab)
# ═══════════════════════════════════════════════════════════
def setup_dependencies():
    """Auto-install missing packages so this works on a fresh Colab T4 runtime."""
    required = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "onnxruntime": "onnxruntime-gpu",
        "tensorrt": "tensorrt",
        "onnx": "onnx",
    }
    missing = []
    for import_name, pkg_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)

    if missing:
        print(f"[INFO] Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        # Re-import after install
        for import_name in missing:
            try:
                __import__(import_name.replace("-", "_").split("==")[0].split(">=")[0])
            except Exception:
                pass

setup_dependencies()

import cv2
import numpy as np
import onnxruntime as ort

# ═══════════════════════════════════════════════════════════
# 2. ONNX → TensorRT Engine Conversion
# ═══════════════════════════════════════════════════════════
def find_trtexec():
    """Find the trtexec binary in Colab/common paths."""
    # 1) Check system PATH
    for cmd in [["which", "trtexec"], ["where", "trtexec"]]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().splitlines()[0]
        except Exception:
            continue

    # 2) Check known Colab / TensorRT install locations
    candidates = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/lib/python3.12/dist-packages/tensorrt_libs/trtexec",
        "/usr/local/lib/python3.11/dist-packages/tensorrt_libs/trtexec",
        "/usr/local/lib/python3.10/dist-packages/tensorrt_libs/trtexec",
        "/usr/local/lib/python3.9/dist-packages/tensorrt_libs/trtexec",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def convert_onnx_to_engine(onnx_path, engine_path, fp16=True, workspace_mb=4096):
    """
    Convert ONNX to TensorRT engine.
    Tries trtexec first; falls back to TensorRT Python API if trtexec is missing.
    Optimized for Google Colab T4 (FP16, ~4 GB workspace).
    """
    if os.path.exists(engine_path):
        print(f"[INFO] Engine already exists: {engine_path}")
        return engine_path

    # ── Method 1: trtexec (fastest, most optimized) ──
    trtexec = find_trtexec()
    if trtexec:
        print(f"[INFO] Found trtexec: {trtexec}")
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--workspace={workspace_mb}",
        ]
        if fp16:
            cmd.append("--fp16")

        print(f"[INFO] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0 and os.path.exists(engine_path):
            print(f"[INFO] trtexec conversion successful: {engine_path}")
            return engine_path
        else:
            print(f"[WARN] trtexec failed, stderr:\n{result.stderr}")

    # ── Method 2: TensorRT Python API (fallback) ──
    print("[INFO] Falling back to TensorRT Python API...")
    import tensorrt as trt
    import onnx

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    # EXPLICIT_BATCH is required for ONNX models
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()

    # T4 has 16 GB VRAM; 4 GB workspace is safe
    if hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_mb * 1024 * 1024

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # YOLOv8n input: [1, 3, 640, 640]
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        min=(1, 3, 640, 640),
        opt=(1, 3, 640, 640),
        max=(1, 3, 640, 640),
    )
    config.add_optimization_profile(profile)

    print("[INFO] Building engine (may take 2–5 min on T4)...")
    # Handle API differences between TensorRT 8.x and 10.x
    if hasattr(builder, "build_engine"):
        engine = builder.build_engine(network, config)          # TRT 8.x
    elif hasattr(config, "build_engine"):
        engine = config.build_engine(network)                   # TRT 10.x
    else:
        serialized = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)

    if engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"[INFO] Engine saved: {engine_path}")
    return engine_path


# ═══════════════════════════════════════════════════════════
# 3. Inference Config & COCO Classes
# ═══════════════════════════════════════════════════════════
MODEL_PATH   = "yolov8n.onnx"
ENGINE_PATH  = "yolov8n_fp16.engine"   # <-- Generated by convert_onnx_to_engine()
INPUT_SIZE   = 640
CONF_THRESH  = 0.25

CLASSES = [
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


# ═══════════════════════════════════════════════════════════
# 4. Model Loading (ONNX Runtime + TensorRT EP)
# ═══════════════════════════════════════════════════════════
def load_model(onnx_path):
    """
    Load ONNX model with ONNX Runtime's TensorRT Execution Provider.
    On first run, ORT builds/optimizes a TensorRT engine internally
    and caches it to disk for faster subsequent starts.
    """
    trt_ep_options = {
        "device_id": 0,                         # T4 GPU ID
        "trt_fp16_enable": True,                # Enable FP16
        "trt_engine_cache_enable": True,        # Cache built engine
        "trt_engine_cache_path": "./trt_cache",
    }
    providers = [
        ("TensorrtExecutionProvider", trt_ep_options),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"[INFO] Active providers: {session.get_providers()}")

    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] Model loaded — Input: {input_name}, Output: {output_name}")
    return session, input_name, output_name


# ═══════════════════════════════════════════════════════════
# 5. Pre / Post Processing
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# 6. Multi-Camera Streaming
# ═══════════════════════════════════════════════════════════
STREAMS = [
    "http://user7:rangsit1025@118.174.138.142:1025/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
    "http://user7:rangsit1033@118.174.138.142:1033/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
    "http://user7:rangsit1030@118.174.138.142:1030/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
    "http://user7:rangsit1031@118.174.138.142:1031/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
    "http://user7:rangsit1035@118.174.138.142:1035/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
    "http://user7:rangsit1029@118.174.138.142:1029/stw-cgi/video.cgi?msubmenu=stream&action=view&Profile=1",
]
STREAM_LABELS = ["Cam1", "Cam2", "Cam3", "Cam4", "Cam5", "Cam6"]
GRID_COLS = 3
GRID_W    = 640
GRID_H    = 360


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
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()


def run_multi(session, input_name, output_name):
    streams = [CameraStream(url, label)
               for url, label in zip(STREAMS, STREAM_LABELS)]

    print("[INFO] Connecting to streams... Press 'q' to quit.")

    while True:
        frames = []
        for stream in streams:
            frame = stream.get_frame()

            if frame is None:
                placeholder = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"{stream.label}: Connecting...",
                            (10, GRID_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (100, 100, 100), 2)
                frames.append(placeholder)
                continue

            # Detection
            orig_h, orig_w = frame.shape[:2]
            inp = preprocess(frame)
            outputs = session.run([output_name], {input_name: inp})
            boxes, scores, class_ids = postprocess_with_nms(outputs, orig_h, orig_w)
            frame = draw(frame, boxes, scores, class_ids)

            cv2.putText(frame, stream.label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            frame = cv2.resize(frame, (GRID_W, GRID_H))
            frames.append(frame)

        # 3×2 grid
        rows = []
        for i in range(0, len(frames), GRID_COLS):
            row_frames = frames[i:i + GRID_COLS]
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


# ═══════════════════════════════════════════════════════════
# 7. Main Entry Point
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Step A: Convert ONNX → TensorRT Engine (Colab T4)
    #         The generated file is ENGINE_PATH ("yolov8n_fp16.engine")
    convert_onnx_to_engine(MODEL_PATH, ENGINE_PATH, fp16=True, workspace_mb=4096)

    # Step B: Load model with ONNX Runtime + TensorRT EP and run inference
    session, input_name, output_name = load_model(MODEL_PATH)
    run_multi(session, input_name, output_name)