import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import torch

from backend.app.detection.pipeline import TrafficAnalysisPipeline


def benchmark(paths, threads):
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    with TemporaryDirectory(prefix="traffic-benchmark-") as directory:
        pipeline = TrafficAnalysisPipeline(evidence_dir=directory, device="cpu")
        for index, path in enumerate(paths):
            capture = cv2.VideoCapture(str(path))
            try:
                success, first_frame = capture.read()
            finally:
                capture.release()
            if not success:
                raise ValueError(f"Could not decode {path.name}")
            pipeline.vehicle_detector.detect(first_frame)
            pipeline.helmet_detector.analyze_head_region(first_frame, dict(x1=0, y1=0, x2=64, y2=64))
            inference_seconds = [0.0]

            def timed(method):
                def invoke(*args, **kwargs):
                    started = perf_counter()
                    try:
                        return method(*args, **kwargs)
                    finally:
                        inference_seconds[0] += perf_counter() - started
                return invoke

            vehicle = pipeline.vehicle_detector
            helmet = pipeline.helmet_detector
            helmet_method = "analyze_heads" if hasattr(helmet, "analyze_heads") else "analyze_head_region"
            detect, analyze = vehicle.detect, getattr(helmet, helmet_method)
            vehicle.detect, wrapped = timed(detect), timed(analyze)
            setattr(helmet, helmet_method, wrapped)
            try:
                result = pipeline.analyze_video(str(path), {"enabled_violations": {
                    "no_helmet": True, "multiple_riders": False, "wrong_way": False,
                }}, f"BENCH-{index}")
            finally:
                vehicle.detect = detect
                setattr(helmet, helmet_method, analyze)
            frames = result["processed_frames"]
            print(json.dumps(dict(video=path.name, frames=frames, fps=result["processing_fps"],
                                  mean_detector_ms=round(inference_seconds[0] * 1000 / frames, 2),
                                  model_available=helmet.is_available,
                                  pending_helmet_incidents=result["summary"]["no_helmet_count"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    benchmark(args.videos, args.threads)
