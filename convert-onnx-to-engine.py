# -*- coding: utf-8 -*-
"""Convert ONNX model to TensorRT .engine file on Jetson Nano.

This script is designed to run directly on the Jetson Nano (not Colab).
TensorRT .engine files are hardware-specific and must be built on the target device.

Examples:
    # Interactive mode (prompts for ONNX path)
    python3 convert-onnx-to-engine.py

    # Direct CLI usage -- YOLO detector
    python3 convert-onnx-to-engine.py \
        --onnx exports/yolo26n_opset12.onnx \
        --engine exports/yolo26n_fp16.engine \
        --imgsz 640

    # Direct CLI usage -- EfficientNet classifier
    python3 convert-onnx-to-engine.py \
        --onnx exports/efficientnetb0_brand_opset12.onnx \
        --engine exports/efficientnetb0_brand_fp16.engine \
        --imgsz 224
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ============================================================
# 0. ONNX pre-processing (opset downgrade + simplify)
# ============================================================
MAX_COMPATIBLE_OPSET = 12


def preprocess_onnx(onnx_path: str) -> str:
    """
    Pre-process an ONNX model for TensorRT 8.x compatibility.

    1) Downgrade opset to 12 (TRT 8.x fails on Gather nodes in opset >=17).
    2) Try onnxsim to simplify the graph (often resolves Gather/Resize issues).
    3) Cast INT64 initialisers to INT32 (TRT clamps INT64 with warnings/errors).
    4) Save to a temp file and return its path.

    If preprocessing is not needed or onnx is not installed, returns the
    original path unchanged.
    """
    try:
        import onnx
        from onnx import TensorProto
    except ImportError:
        print("[WARN] onnx package not installed; skipping ONNX preprocessing.")
        print("[WARN] Install with:  pip install onnx onnxsim")
        return onnx_path

    model = onnx.load(onnx_path)
    original_opset = getattr(model.opset_import[0], "version", None)
    needs_fix = False

    if original_opset is not None and original_opset > MAX_COMPATIBLE_OPSET:
        print(f"[INFO] Opset {original_opset} > {MAX_COMPATIBLE_OPSET}; downgrading for TRT 8.x compatibility.")
        needs_fix = True
    else:
        print(f"[INFO] Opset version: {original_opset}")

    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT64:
            needs_fix = True
            break

    if not needs_fix:
        try:
            onnxsim = __import__("onnxsim")
            print("[INFO] Running onnxsim to simplify model graph...")
            model, success = onnxsim.simplify(model)
            if success:
                print("[INFO] onnxsim simplification succeeded.")
            else:
                print("[WARN] onnxsim simplification reported issues; continuing with original.")
                model = onnx.load(onnx_path)
        except ImportError:
            print("[INFO] onnxsim not installed; skipping graph simplification.")

        return onnx_path

    # --- Apply fixes ---
    # Downgrade opset
    if original_opset is not None and original_opset > MAX_COMPATIBLE_OPSET:
        model.opset_import[0].version = MAX_COMPATIBLE_OPSET

    # Cast INT64 initializers to INT32
    import numpy as np
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT64:
            arr = onnx.numpy_helper.to_array(init)
            arr = arr.astype(np.int32)
            new_init = onnx.numpy_helper.from_array(arr, init.name)
            init.CopyFrom(new_init)

    # Also fix INT64 constant nodes
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.t.data_type == TensorProto.INT64:
                    arr = onnx.numpy_helper.to_array(attr.t)
                    arr = arr.astype(np.int32)
                    attr.t.CopyFrom(onnx.numpy_helper.from_array(arr))

    # Try onnxsim simplification after fixes
    try:
        onnxsim = __import__("onnxsim")
        print("[INFO] Running onnxsim after opset downgrade + INT64 cast...")
        model, success = onnxsim.simplify(model)
        if success:
            print("[INFO] onnxsim simplification succeeded.")
        else:
            print("[WARN] onnxsim reported issues; using fixed-but-unsimplified model.")
    except ImportError:
        print("[INFO] onnxsim not installed; skipping simplification after fixes.")

    # Validate before saving
    try:
        onnx.checker.check_model(model, full_check=False)
        print("[INFO] ONNX model validation passed after preprocessing.")
    except Exception as e:
        print(f"[WARN] ONNX validation issue (may still work): {e}")

    # Save to temp file next to the original
    tmp_path = str(Path(onnx_path).with_suffix(".opset12.onnx"))
    onnx.save(model, tmp_path)
    print(f"[INFO] Saved preprocessed ONNX: {tmp_path}")
    return tmp_path


# ============================================================
# 1. Locate trtexec
# ============================================================
def find_trtexec():
    """Find the trtexec binary. Prioritizes Jetson paths, then system PATH."""
    # 1) Known Jetson / TensorRT install locations
    candidates = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
        "/usr/bin/trtexec",
    ]
    # 2) JetPack Python wheel paths
    for py_ver in ("python3.12", "python3.11", "python3.10", "python3.9", "python3.8"):
        candidates.append(
            f"/usr/local/lib/{py_ver}/dist-packages/tensorrt_libs/trtexec"
        )
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    # 3) Check system PATH
    for cmd in (["which", "trtexec"], ["where", "trtexec"]):
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().splitlines()[0]
        except Exception:
            continue
    return None


# ============================================================
# 2. ONNX -> TensorRT Engine Conversion
# ============================================================
def convert_onnx_to_engine(
    onnx_path: str,
    engine_path: str,
    imgsz: int = 640,
    fp16: bool = True,
    workspace_mb: int = 1024,
    dynamic: bool = False,
    batch: int = 1,
) -> str:
    """
    Convert ONNX to TensorRT engine.
    Tries trtexec first; falls back to TensorRT Python API if trtexec is missing.
    Optimized for Jetson Nano (conservative workspace, FP16).
    """
    onnx_path_orig = Path(onnx_path)
    if not onnx_path_orig.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path_orig}")

    # Pre-process ONNX for TRT 8.x compatibility (opset downgrade, INT64→INT32)
    preprocessed = preprocess_onnx(str(onnx_path_orig))
    onnx_path = Path(preprocessed)
    engine_path = Path(engine_path)

    if engine_path.exists():
        print(f"[INFO] Engine already exists: {engine_path}")
        return str(engine_path)

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Method 1: trtexec (fastest, most optimized) --
    trtexec = find_trtexec()
    if trtexec:
        print(f"[INFO] Found trtexec: {trtexec}")
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--workspace={workspace_mb}",
            f"--minShapes=images:{batch}x3x{imgsz}x{imgsz}",
            f"--optShapes=images:{batch}x3x{imgsz}x{imgsz}",
            f"--maxShapes=images:{batch}x3x{imgsz}x{imgsz}",
        ]
        if fp16:
            cmd.append("--fp16")

        print(f"[INFO] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=None, stderr=None, universal_newlines=True)
        if result.returncode == 0 and engine_path.exists():
            print(f"[INFO] trtexec conversion successful: {engine_path}")
            return str(engine_path)
        else:
            print(f"[WARN] trtexec failed (rc={result.returncode}).")
            if result.stderr:
                print(f"[WARN] stderr:\n{result.stderr}")

    # -- Method 2: TensorRT Python API (fallback) --
    print("[INFO] Falling back to TensorRT Python API...")
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT Python API not available. "
            "Install TensorRT for JetPack or ensure trtexec is on PATH."
        ) from exc

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

    # Jetson Nano has only 4 GB shared memory; keep workspace modest
    if hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_mb * 1024 * 1024

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()

    if dynamic:
        # Allow batch size to vary between 1 and user-provided batch
        max_batch = max(batch, 1)
        profile.set_shape(
            input_tensor.name,
            min=(1, 3, imgsz, imgsz),
            opt=(1, 3, imgsz, imgsz),
            max=(max_batch, 3, imgsz, imgsz),
        )
    else:
        profile.set_shape(
            input_tensor.name,
            min=(batch, 3, imgsz, imgsz),
            opt=(batch, 3, imgsz, imgsz),
            max=(batch, 3, imgsz, imgsz),
        )
    config.add_optimization_profile(profile)

    print(f"[INFO] Building engine (imgsz={imgsz}, batch={batch}, fp16={fp16})...")
    print("[INFO] This may take several minutes on Jetson Nano.")

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
    return str(engine_path)


# ============================================================
# 3. Interactive prompt helper
# ============================================================
def prompt_for_onnx_path() -> str:
    """Prompt user for an ONNX file path if --onnx was not provided."""
    print("No --onnx argument provided. Enter the path to your ONNX file.")
    while True:
        raw = input("ONNX path: ").strip()
        if not raw:
            print("Path cannot be empty. Try again.")
            continue
        p = Path(raw)
        if not p.exists():
            print(f"File not found: {p}. Try again.")
            continue
        if p.suffix.lower() != ".onnx":
            print(f"Warning: file does not end with .onnx ({p.suffix}). Proceeding anyway.")
        return str(p)


# ============================================================
# 4. Argument parser
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ONNX to TensorRT .engine on Jetson Nano."
    )
    parser.add_argument(
        "--onnx", "-i",
        default=None,
        help="Path to input ONNX model. If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--engine", "-o",
        default=None,
        help="Path to output .engine file. Defaults to <input>_fp16.engine next to the ONNX file.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Model input size (square). Default 640 for YOLO; use 224 for EfficientNet.",
    )
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument(
        "--fp16", action="store_true", default=True,
        help="Enable FP16 precision (default).",
    )
    fp16_group.add_argument(
        "--no-fp16", action="store_true", default=False,
        help="Disable FP16 precision.",
    )
    parser.add_argument(
        "--workspace", type=int, default=1024,
        help="TensorRT workspace in MB. Default 1024 for Jetson Nano safety.",
    )
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Allow dynamic batch size (min=1, max=--batch).",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Batch size for static shapes, or max batch for dynamic. Default 1.",
    )
    return parser.parse_args()


# ============================================================
# 5. Main entry point
# ============================================================
def main() -> None:
    args = parse_args()

    # Resolve ONNX path
    onnx_path = args.onnx
    if onnx_path is None:
        onnx_path = prompt_for_onnx_path()
    else:
        onnx_path = str(Path(onnx_path))
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    fp16 = not args.no_fp16

    # Resolve engine path
    if args.engine:
        engine_path = str(Path(args.engine))
    else:
        p = Path(onnx_path)
        suffix = "_fp16.engine" if fp16 else ".engine"
        engine_path = str(p.parent / (p.stem + suffix))

    print(f"[INFO] Input ONNX : {onnx_path}")
    print(f"[INFO] Output engine: {engine_path}")
    print(f"[INFO] imgsz={args.imgsz} | fp16={fp16} | workspace={args.workspace}MB | batch={args.batch} | dynamic={args.dynamic}")

    convert_onnx_to_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        imgsz=args.imgsz,
        fp16=fp16,
        workspace_mb=args.workspace,
        dynamic=args.dynamic,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
