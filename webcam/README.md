# Webcam recognition toolkit (first milestone)

Makes the frozen INCLUDE262 model usable on real webcam input **without retraining**:
active-sign trimming, a camera quality gate, confidence/margin rejection, exact
training-preprocessing reuse, a labelled data collector, and a webcam benchmark.

Nothing here touches the locked INCLUDE test evaluation, the original checkpoint,
`scripts/`, `datasets/`, `metadata/*` or `processed_data/INCLUDE262/*`.

All commands use the project venv: `./venv/Scripts/python.exe`.

---

## 1. Check whether a sign is supported (closed set of 262)

```
./venv/Scripts/python.exe -m webcam.vocab "thank you"      # -> class 223 thankyou
./venv/Scripts/python.exe -m webcam.vocab "helicopter"     # -> outside the 262-sign vocabulary
./venv/Scripts/python.exe -m webcam.vocab --list           # all 262
./venv/Scripts/python.exe -m webcam.vocab --check goodnight # exit 0 if supported
```

Backend: `GET /vocab?q=hello`. Frontend: the "Supported vocabulary" card.

---

## 2. Diagnose webcam vs training preprocessing

```
./venv/Scripts/python.exe -m webcam.diagnose_preprocessing \
    --include hello --webcam webcam_data/val/c101_hello_s1_....mp4 --label hello
```

`--include` accepts a file path, a label, or an integer `source_index` into
`metadata/include262_official_split_final.csv` (train/val only — never test).
Writes `results/webcam_preprocessing_diagnostic_<ts>.txt` + `.json` with a
stage-by-stage table and a flagged delta report.

## 3. Diagnose mirroring / handedness

```
./venv/Scripts/python.exe -m webcam.diagnose_handedness \
    --webcam webcam_data/val/c101_hello_s1_....mp4 --label hello
```

Runs inference on the raw clip and a horizontally-flipped copy. Writes
`results/webcam_handedness_diagnostic_<ts>.txt` with an explicit verdict.

---

## 4. Collect labelled webcam samples (dev-only)

Start the backend with the collector enabled, and the frontend in collector mode:

```
# terminal 1 — backend
INCLUDE262_COLLECTOR=1 ./venv/Scripts/python.exe -m uvicorn main:app --app-dir app/backend --port 8000

# terminal 2 — frontend
cd app/frontend && VITE_COLLECTOR=1 npm run dev
```

Open http://localhost:5173 → **Data collection** tab:

1. Search and select a supported sign (shows class_id + label).
2. Set `session_id`, background, lighting, position, camera distance, clothing.
3. Choose **train** or **val**.
4. Start camera → Record sign → Stop → preview.
5. **Save to train/val**, or Re-record.
6. The per-class / per-split tally updates after each save.

Clips + metadata sidecars land in `webcam_data/{train,val}/`; a row is appended
to `webcam_data/manifest.csv`. The label you picked is ground truth — the model
is never used to label.

**Collection strategy** (Modules 11, 15): pick ~20–30 common signs
(`hello`, `thankyou`, `howareyou`, `goodmorning`, `goodnight`, `friend`, `family`,
`mother`, `father`, `brother`, `sister`, `house`, `school`, `hospital`, `doctor`,
`teacher`, `student`, `book`, `money`, `time`, `today`, `tomorrow`, `yesterday`,
`good`, `bad`, `happy`, `sad`, `hot`, `cold`, `you` — validate with `webcam.vocab`).
Collect **8–12 train + 3–5 val** clips per sign across **≥2 sessions**, varying
background / lighting / clothing / distance. Keep **train and val on different
sessions/backgrounds** so validation measures generalization, not memorization.

---

## 5. Preprocess the collected clips

```
./venv/Scripts/python.exe -m webcam.preprocess_webcam            # all splits, resumable
./venv/Scripts/python.exe -m webcam.preprocess_webcam --no-trim  # ablation: skip trimming
```

Writes `processed_data/WEBCAM/model_features64_v1/<split>/*.npz` and
`webcam_data/model_features_manifest.csv`. Uses the **frozen INCLUDE TRAIN
normalization** only.

---

## 6. Run the webcam baseline (original checkpoint)

```
./venv/Scripts/python.exe -m webcam.evaluate_webcam_baseline
```

Reports **WEBCAM VALIDATION ACCURACY** (Top-1/Top-5, macro-F1, per-class,
confusions, mean confidence, rejection rate, accepted-prediction accuracy, and a
by-background / by-lighting / by-session breakdown). Writes
`results/webcam_baseline_<ts>_*`. This is a separate metric from the locked
INCLUDE test result (98.04%) and must never be reported as such.

Optional attribution + distribution checks:

```
./venv/Scripts/python.exe -m webcam.evaluate_webcam_baseline --split train   # sanity on train clips
./venv/Scripts/python.exe -m webcam.feature_stats                            # INCLUDE vs webcam distributions
```

---

---

## 7b. Webcam domain adaptation (only after the baseline confirms a real gap)

The original checkpoint is the initialization; it is **never overwritten**.

```
# 1. collect ~20-30 classes x (10-15 train / 4-5 val) with the Data collection tab,
#    varying background / lighting / clothing / distance / speed across >=2 sessions.
# 2. preprocess (frozen INCLUDE train normalization only)
./venv/Scripts/python.exe -m webcam.preprocess_webcam

# 3. (optional) if MediaPipe coverage is bad for some backgrounds, add mild visual variants:
./venv/Scripts/python.exe -m webcam.visual_augment --variants 2
./venv/Scripts/python.exe -m webcam.preprocess_webcam        # re-run to pick them up

# 4. baseline the original model on webcam VAL
./venv/Scripts/python.exe -m webcam.evaluate_webcam_baseline

# 5. adapt: init from ema_state, INCLUDE train + webcam train, balanced sampling,
#    Stage A (encoders frozen) -> Stage B (unfrozen), feature-space augmentation,
#    best webcam-val checkpoint kept only if INCLUDE-val top-1 stays within 2 pts.
./venv/Scripts/python.exe -m webcam.adapt_train
#   tunables: --webcam-frac 0.5 --epochs-a 8 --epochs-b 14 --lr-a 3e-4 --lr-b 4e-5
#             --include-tolerance 2.0 --no-aug   (--smoke for a 2+2 epoch dry run)
#   -> models/checkpoints/include262_webcam_adapted_v1.pt  (+ results/webcam_adapt_v1_*)

# 6. compare original vs adapted on the SAME webcam VAL + INCLUDE VAL, per condition
./venv/Scripts/python.exe -m webcam.compare_models

# 7. deploy the adapted model for the Live camera path ONLY IF it clearly improved:
INCLUDE262_MODEL_VARIANT=webcam_adapted ./venv/Scripts/python.exe -m uvicorn main:app --app-dir app/backend --port 8000
#   /predict (Upload) always stays on the original checkpoint.
#   GET /model-info shows the active live variant.
```

Feature-space augmentation lives in `webcam/augment.py` (`AugConfig` defaults):
coordinate jitter, global translation/scale, temporal speed warp, frame/column
dropout, modality dropout (never both hands), velocity/acceleration perturbation.
**No horizontal flip** — the part-aware encoders are left/right-asymmetric.

---

## Live camera path

The **Live camera** tab now calls `POST /predict/live`:

```
active-sign trim  ->  quality gate  ->  exact training preprocessing (scripts 76/79)
   ->  frozen INCLUDE262 model  ->  top-1 confidence + top1/top2 margin rejection
```

Starting rejection thresholds (dev only — tune on webcam val, never on the locked
test): `min_confidence = 0.70`, `min_margin = 0.15` (`app/backend/rejection.py`).
Trim / quality thresholds live in `app/backend/trim.py` and `app/backend/quality.py`.

The frozen `POST /predict` (Upload tab) is unchanged.
