# Helmet backend audit and evaluation

- A detected person could previously be assigned to several overlapping motorcycles. Association now picks only a clear best match; a motorcycle can still have multiple riders.
- Helmet inference previously picked the strongest label anywhere in a padded crop. It now matches boxes to a rider's head, abstains on conflicting or shared ownership, and batches crops. Video alerts require three consecutive no-helmet observations for the same rider and motorcycle. All decoded frames are analyzed rather than skipping short incidents.
- One-frame alerts, reused stale wrong-way tracks, unbounded detection and trajectory history, and unverified video/evidence writes were corrected. An unavailable vehicle model fails the job instead of producing a successful empty result. Both checkpoints are unchanged; accelerator inference can fall back to CPU.
- The local FastAPI demo still runs on loopback. Hosted incident review and evidence now use authenticated Netlify routes, managed Postgres and private Blobs. The Python CLI runs OpenCV/PyTorch outside the request runtime.

## Measured results

Bundled clips, same checkpoints, CPU with two PyTorch threads, inference warmed up, annotated video written, other rules disabled. FPS includes decode, inference, tracking, and annotation; detector time is mean vehicle plus helmet time per input frame. Counts are pending alerts, not correct detections.

| Clip | Frames | Baseline → revised FPS | Baseline → revised detector ms/frame | Pending alerts |
| --- | ---: | ---: | ---: | ---: |
| `motorcycle_helmet_violation.mp4` | 60 | 19.40 → 19.41 | 36.18 → 36.02 | 1 → 0 |
| `motorcycle_traffic_1.mp4` | 60 | 19.39 → 21.17 | 36.04 → 32.46 | 1 → 0 |
| `motorcycle_traffic_2.mp4` | 60 | 21.35 → 22.32 | 32.90 → 30.60 | 1 → 0 |
| `traffic_highway.mp4` (back-to-back rerun) | 601 | 31.22 → 36.89 | 26.84 → 22.17 | 0 → 0 |

Highway runs varied: an earlier run measured 38.52 → 13.62 FPS, so the rerun is not proof of a repeatable speedup. The baseline evidence on the helmet demo showed an apparently helmeted rider at 0.33 s; this is one qualitative observation, not a false-positive rate. The duplicate demo video was excluded. A downscaled, darkened, and blurred copy of `motorcycle_traffic_1.mp4` processed all 60 frames at 25.94 FPS (37.01 ms detector time) with zero alerts. This is a degradation smoke test, not a night-time validation set.

Reproduce with `python -m ml.evaluation.benchmark_video sample_media/motorcycle_helmet_violation.mp4 sample_media/motorcycle_traffic_1.mp4 sample_media/motorcycle_traffic_2.mp4 sample_media/traffic_highway.mp4`. Use the same script from a baseline checkout with that checkout on `PYTHONPATH`.

**Accuracy is unverified:** the clips have no usable per-rider helmet ground truth. Precision, recall, F1, false-positive rate, and false-negative rate cannot be computed. No positive helmet alert was observed in the revised real-clip smoke test. The consecutive-frame rule can miss brief, occluded, and distant riders. Tracking can still switch IDs after sustained occlusion. Label representative clips before tuning thresholds, widening inference resolution, or validating night-time/crowd performance. Sustained stream memory and GPU behavior were not measured.

Automated checks covered competing bike ownership, adjacent riders, occlusion, a 3,000-frame tracker regression, one scripted passenger incident, corrupt video, immutable evidence, reviewer authorization, idempotency, and persisted Postgres decisions after reopening the database. The scripted incident tests orchestration, not model accuracy. Live Identity, Database, and Blobs were not exercised together on a deployed site.
