"""
Standalone INCLUDE262 inference sanity check.

Usage:
    python app/backend/test_inference.py "path/to/video.mov"

Prints the three preprocessing tensor shapes (for verification) and the
model prediction (sign, confidence, top-5, modality coverage, timings).
No FastAPI. No training. No writes to existing project folders.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _fmt_pct(x):
    return f"{100.0 * x:6.2f}%"


def main(argv):
    if len(argv) != 2:
        print('Usage: python app/backend/test_inference.py "path/to/video.mov"')
        return 2

    video_path = argv[1]
    if not os.path.isfile(video_path):
        print(f"ERROR: file not found: {video_path}")
        return 2

    import config
    import inference

    try:
        config.check_paths()
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print("=" * 78)
    print("INCLUDE262 STANDALONE INFERENCE SANITY CHECK")
    print("=" * 78)
    print(f"Video      : {os.path.abspath(video_path)}")
    print(f"Checkpoint : {config.CHECKPOINT.name}  (ema_state)")
    print(f"Device     : {config.DEVICE}")
    print("Running MediaPipe extraction + feature build + model forward ...")
    print()

    result = inference.predict(video_path, topk=5)

    shapes = result["tensor_shapes"]
    print("-" * 78)
    print("PREPROCESSING TENSOR SHAPES")
    print("-" * 78)
    checks = [
        ("landmark", shapes["landmark"], [config.NUM_FRAMES, config.LANDMARK_DIM]),
        ("geometry", shapes["geometry"], [config.NUM_FRAMES, config.GEOMETRY_DIM]),
        ("availability", shapes["availability"], [config.NUM_FRAMES, config.AVAILABILITY_DIM]),
    ]
    all_ok = True
    for name, got, want in checks:
        ok = list(got) == want
        all_ok &= ok
        print(f"  {name:<13}: {str(tuple(got)):<14} expected {str(tuple(want)):<14} "
              f"[{'OK' if ok else 'MISMATCH'}]")
    print(f"  shape contract: {'PASS' if all_ok else 'FAIL'}")
    print()

    print("-" * 78)
    print("PREDICTION")
    print("-" * 78)
    print(f"  predicted_class_id : {result['predicted_class_id']}")
    print(f"  predicted_sign     : {result['predicted_sign']}")
    print(f"  confidence         : {result['confidence']:.4f}")
    print()
    print("  top-5:")
    print(f"    {'rank':<5}{'class_id':<10}{'sign':<24}{'probability'}")
    for entry in result["top5"]:
        print(f"    {entry['rank']:<5}{entry['class_id']:<10}{entry['sign']:<24}"
              f"{entry['probability']:.4f}")
    print()

    cov = result["modality_coverage"]
    print("-" * 78)
    print("MODALITY COVERAGE")
    print("-" * 78)
    pf = cov["per_frame_availability"]
    print("  per-frame availability (fraction of 64 frames present):")
    for key in ("left_hand", "right_hand", "pose", "face"):
        print(f"    {key:<12}: {_fmt_pct(pf[key])}")
    er = cov["extractor_rates"]
    print("  extractor rates (script 76):")
    for key in ("left_final_rate", "right_final_rate", "both_hands_final_rate",
                "left_rescue_rate", "right_rescue_rate", "pose_rate", "face_rate"):
        print(f"    {key:<22}: {_fmt_pct(er[key])}")
    print()

    print("-" * 78)
    print("TIMING")
    print("-" * 78)
    print(f"  preprocessing : {result['preprocess_seconds']:.2f} s")
    print(f"  model forward  : {result['inference_seconds'] * 1000:.1f} ms")
    print(f"  extraction status: {result['extraction_status']}")
    print(f"  checkpoint epoch : {result['checkpoint']['epoch']}")
    print("=" * 78)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
