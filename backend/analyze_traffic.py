import argparse
import hashlib
import json
import logging
import os
from contextlib import ExitStack
from getpass import getpass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from backend.app.detection.pipeline import TrafficAnalysisPipeline

logger = logging.getLogger(__name__)


def send_incident(client, incident, source, digest):
    identity = [source, digest, incident["camera_id"], incident["violation_type"], incident["tracking_id"]]
    fields = ("camera_id", "tracking_id", "motorcycle_tracking_id", "violation_type", "timestamp", "frame_number",
              "detection_confidence", "violation_confidence", "explanation")
    metadata = {field: incident[field] for field in fields if field in incident}
    metadata.update(source=source, incident_key=hashlib.sha256(json.dumps(identity).encode()).hexdigest())
    for attempt in range(3):
        try:
            with ExitStack() as opened:
                files = {name: (f"{name}.jpg", opened.enter_context(open(incident[field], "rb")), "image/jpeg")
                         for name, field in (("original", "evidence_path"), ("crop", "crop_path"),
                                             ("annotated", "annotated_evidence_path"))}
                response = client.post("/api/violations/ingest", data={"metadata": json.dumps(metadata)}, files=files)
            if response.status_code < 500 and response.status_code != 429:
                if response.status_code in (200, 201):
                    return
                raise RuntimeError(f"Evidence submission rejected (HTTP {response.status_code})")
        except httpx.TransportError:
            pass
        if attempt < 2:
            sleep(2 ** attempt)
    raise RuntimeError("Could not persist the incident after three attempts; rerun the video to retry")


def main():
    parser = argparse.ArgumentParser(description="Analyze a traffic video and submit pending incidents to Netlify.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--site", required=True, help="HTTPS origin of the deployed review API")
    parser.add_argument("--source", required=True, help="Non-sensitive logical name of the recording")
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--device", default=None, help="Override accelerator selection, e.g. cpu")
    parser.add_argument("--helmet-confidence", type=float, default=0.50)
    options = parser.parse_args()
    site = urlsplit(options.site)
    if (site.scheme != "https" or not site.hostname or site.username or site.password or site.query or site.fragment
            or site.path not in ("", "/")):
        parser.error("--site must be an HTTPS origin without user information or a path")
    if not 0 < options.helmet_confidence <= 1:
        parser.error("--helmet-confidence must be in (0, 1]")
    if not options.video.is_file():
        parser.error("Video file not found")

    token = os.getenv("NETLIFY_IDENTITY_TOKEN") or getpass("Ingest account access token: ")
    with options.video.open("rb") as video:
        digest = hashlib.file_digest(video, "sha256").hexdigest()
    with TemporaryDirectory(prefix="traffic-evidence-") as directory, httpx.Client(
        base_url=options.site, headers={"Authorization": f"Bearer {token}"}, timeout=30, follow_redirects=False,
    ) as client:
        pipeline = TrafficAnalysisPipeline(evidence_dir=directory, device=options.device,
                                           helmet_confidence=options.helmet_confidence)
        if not pipeline.vehicle_detector.is_loaded or not pipeline.helmet_detector.is_available:
            raise RuntimeError("Vehicle and helmet model checkpoints must load before analysis")

        def persist(incident):
            send_incident(client, incident, options.source, digest)
            for field in ("evidence_path", "crop_path", "annotated_evidence_path"):
                Path(incident[field]).unlink()

        result = pipeline.analyze_video(
            str(options.video), {"id": options.camera, "enabled_violations": {
                "no_helmet": True, "multiple_riders": True, "wrong_way": False,
            }}, f"JOB-{uuid4().hex}", incident_callback=persist, write_annotated_video=False,
        )
    print(json.dumps({key: result[key] for key in ("processed_frames", "processing_fps", "mean_inference_ms", "summary")}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("Processing failed: %s", error)
        raise SystemExit(1) from None
