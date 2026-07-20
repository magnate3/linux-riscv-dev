import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
from pathlib import Path
import time
import ncnn
from ultralytics import YOLO
import torch
# Hàm benchmark ONNX
def evaluate_onnx(onnx_path, data_yaml, eval_type="Self", device="cpu", imgsz=640, batch=4, num_threads=8, model_name=None):
    print(f"\n📈 [{eval_type}] Evaluating (ONNX): {os.path.basename(onnx_path)} on {data_yaml}")

    # --- Setup threading & provider ---
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]

    # --- Load YOLO ONNX for evaluation ---
    model = YOLO(onnx_path, task="detect")

    # --- Validation phase ---
    t0 = time.time()
    metrics = model.val(data=data_yaml, split="val", imgsz=imgsz, batch=batch, device=device, verbose=False)
    eval_time = time.time() - t0

    # --- Extract metrics ---
    prec = round(metrics.results_dict.get("metrics/precision(B)", 0), 4)
    recall = round(metrics.results_dict.get("metrics/recall(B)", 0), 4)
    map50 = round(metrics.results_dict.get("metrics/mAP50(B)", 0), 4)
    map5095 = round(metrics.results_dict.get("metrics/mAP50-95(B)", 0), 4)
    f1 = round(2 * (prec * recall) / (prec + recall + 1e-9), 4)
    acc = round((prec + recall) / 2, 4)

    # --- Benchmark inference using onnxruntime ---
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)


    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # Warmup + benchmark
    warmup, runs = 5, 20
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy})
    t_inf = time.time()
    for _ in range(runs):
        _ = session.run(None, {input_name: dummy})
    infer_time = (time.time() - t_inf) / runs
    fps = round(1.0 / infer_time, 2)

    return {
        "Model Name": model_name,
        "Backend": "ONNX",
        "Eval Type": eval_type,
        "Precision": prec,
        "Recall": recall,
        "F1-score": f1,
        "Accuracy": acc,
        "mAP@50": map50,
        "mAP@50-95": map5095,
        "Eval Time (s)": round(eval_time, 2),
        "Infer Time (s/img)": round(infer_time, 4),
        "FPS": fps

    }
def evaluate_ncnn(ncnn_folder, data_yaml, eval_type="Self", imgsz=640, num_threads=4, model_name=None):
    """
    Benchmark NCNN model (inference bằng runtime thuần NCNN, val dùng YOLO wrapper)
    """
    print(f"\n📈 [{eval_type}] Evaluating (NCNN): {os.path.basename(ncnn_folder)} on {data_yaml}")

    # --- Load model YOLO (wrapper) để tính mAP ---
    try:
        model = YOLO(ncnn_folder, task="detect")
        t0 = time.time()
        metrics = model.val(data=data_yaml, imgsz=imgsz, split="val", batch=1, device="cpu", verbose=False)
        eval_time = time.time() - t0

        prec = round(metrics.results_dict.get("metrics/precision(B)", 0), 4)
        recall = round(metrics.results_dict.get("metrics/recall(B)", 0), 4)
        map50 = round(metrics.results_dict.get("metrics/mAP50(B)", 0), 4)
        map5095 = round(metrics.results_dict.get("metrics/mAP50-95(B)", 0), 4)
        f1 = round(2 * (prec * recall) / (prec + recall + 1e-9), 4)
        acc = round((prec + recall) / 2, 4)
        print(f"✅ Validation done: mAP@50={map50}")
    except Exception as e:
        print(f"⚠️ Validation failed with YOLO wrapper: {e}")
        prec = recall = map50 = map5095 = f1 = acc = 0
        eval_time = 0

    # --- Benchmark inference bằng NCNN thuần ---
    print("🔧 Benchmarking raw NCNN inference...")
    param_path = Path(ncnn_folder) / "model.ncnn.param"
    bin_path = Path(ncnn_folder) / "model.ncnn.bin"
    if not param_path.exists() or not bin_path.exists():
        print(f"❌ Missing NCNN files in {ncnn_folder}")
        return None

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = num_threads
    net.load_param(str(param_path))
    net.load_model(str(bin_path))

    # --- Dummy input ---
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # --- Warmup ---
    warmup, runs = 5, 20
    for _ in range(warmup):
        ex = net.create_extractor()
        ex.input("in0", ncnn.Mat(dummy))
        _, _ = ex.extract("out0")  

    # --- Benchmark ---
    t_inf = time.time()
    for _ in range(runs):
        ex = net.create_extractor()
        ex.input("in0", ncnn.Mat(dummy))
        _, _ = ex.extract("out0")
    infer_time = (time.time() - t_inf) / runs
    fps = round(1.0 / infer_time, 2)

    print(f"✅ NCNN Inference time: {infer_time*1000:.2f} ms/image")

    return {
        "Model Name": model_name,
        "Backend": "NCNN",
        "Eval Type": eval_type,
        "Precision": prec,
        "Recall": recall,
        "F1-score": f1,
        "Accuracy": acc,
        "mAP@50": map50,
        "mAP@50-95": map5095,
        "Eval Time (s)": round(eval_time, 2),
        "Infer Time (s/img)": round(infer_time, 4),
        "FPS": fps
    }
# --- Đường dẫn dataset duy nhất ---
dataset1 = "/home/pi5/TrafficSign/Dataset/Detect/data.yaml"
models_ncnn = {
    "YOLOv5n": "/home/pi5/TrafficSign/convert/model/yolov5/yolo5_ncnn_model",
    "YOLOv8n": "/home/pi5/TrafficSign/convert/model/yolov8/yolo8_ncnn_model",
    "YOLO11n": "/home/pi5/TrafficSign/convert/model/yolov11/yolo11_ncnn_model",
}

results = []
# --- Benchmark NCNN ---
for model_name, model_path in models_ncnn.items():
    print(f"\n🚀 Running NCNN Benchmark for {model_name}")
    result = evaluate_ncnn(model_path, dataset1, eval_type=f"{model_name}", imgsz=640, num_threads=4, model_name=model_name)
    if result:
        results.append(result)
