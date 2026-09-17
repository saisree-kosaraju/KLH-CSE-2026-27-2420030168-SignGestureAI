"""
Feature-space augmentation for webcam domain adaptation (Module 7 / 13).

Operates on the BUILT model features (landmark 64x534, geometry 64x469,
availability 64x4) — the same tensors scripts/79 produces and scripts/80/82
already train on. The existing training used mild Gaussian noise + frame dropout;
this adds the richer, still-conservative set the adaptation brief asks for.

Nothing here creates anatomically impossible landmark configurations, and there is
NO horizontal flip (handedness semantics are asymmetric in the part-aware model —
`diagnose_handedness` confirmed the un-flipped convention, and flipping would swap
the left/right feature streams the encoders were trained on).

landmark_stream layout (scripts/79 schema):
    0:63    left hand  xyz (21x3, body-normalized + z-scored)
    63:126  right hand xyz
    126:225 pose xyz
    225:258 pose visibility
    258:534 selected face xyz
geometry_stream layout: coords 0:126, bones 126:166, angles 166:196,
    relations 196:217, velocity 217:343, acceleration 343:469
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_FRAMES = 64

# landmark_stream blocks that hold xyz coordinates (translate/scale act on these)
_LM_COORD_BLOCKS = [(0, 63), (63, 126), (126, 225), (258, 534)]
_LM_XY_MASK = None  # built lazily: True where a landmark_stream dim is an x or y coord


def _lm_xy_mask() -> np.ndarray:
    global _LM_XY_MASK
    if _LM_XY_MASK is None:
        m = np.zeros(534, dtype=bool)
        for a, b in _LM_COORD_BLOCKS:
            for i in range(a, b):
                if (i - a) % 3 in (0, 1):  # x or y within the xyz triple
                    m[i] = True
        _LM_XY_MASK = m
    return _LM_XY_MASK


@dataclass(frozen=True)
class AugConfig:
    enabled: bool = True
    # gaussian feature noise
    coord_jitter_std: float = 0.012
    # global spatial translation (z-scored units) applied to all coord blocks
    translate_p: float = 0.6
    translate_std: float = 0.12
    # global spatial scale
    scale_p: float = 0.5
    scale_range: tuple = (0.93, 1.07)
    # temporal speed / resample
    time_warp_p: float = 0.5
    time_warp_range: tuple = (0.85, 1.15)
    # frame (token) dropout
    frame_dropout_p: float = 0.06
    # landmark-column dropout (drop whole feature dims across all frames)
    col_dropout_p: float = 0.5
    col_dropout_frac: float = 0.02
    # modality dropout (never drops both hands)
    modality_dropout_p: float = 0.12
    # velocity / acceleration perturbation (geometry slices)
    motion_perturb_p: float = 0.5
    motion_scale_range: tuple = (0.85, 1.15)
    motion_noise_std: float = 0.02


DEFAULT_AUG = AugConfig()
NO_AUG = AugConfig(enabled=False)


def _time_warp(x: np.ndarray, rate: float, order_nn: bool) -> np.ndarray:
    """Resample the time axis by `rate` then crop/pad back to 64 frames."""
    t = x.shape[0]
    src = np.arange(t, dtype=np.float32)
    dst = np.arange(0, t, rate, dtype=np.float32)
    if order_nn:
        idx = np.clip(np.round(dst).astype(int), 0, t - 1)
        y = x[idx]
    else:
        y = np.stack([np.interp(dst, src, x[:, c]) for c in range(x.shape[1])], axis=1).astype(np.float32)
    if len(y) >= t:
        return y[:t]
    pad = np.repeat(y[-1:], t - len(y), axis=0)
    return np.concatenate([y, pad], axis=0)


def augment(landmark: np.ndarray, geometry: np.ndarray, availability: np.ndarray,
            cfg: AugConfig = DEFAULT_AUG, rng: np.random.Generator | None = None):
    """Return augmented copies of (landmark, geometry, availability). Shapes unchanged."""
    if not cfg.enabled:
        return landmark, geometry, availability
    rng = rng or np.random.default_rng()
    lm = landmark.astype(np.float32).copy()
    gm = geometry.astype(np.float32).copy()
    av = availability.astype(np.float32).copy()

    xy = _lm_xy_mask()

    # --- global spatial translation (keeps relative hand geometry intact) ---
    if rng.random() < cfg.translate_p:
        tx, ty = rng.normal(0.0, cfg.translate_std, size=2).astype(np.float32)
        for a, b in _LM_COORD_BLOCKS:
            for i in range(a, b):
                if (i - a) % 3 == 0:
                    lm[:, i] += tx
                elif (i - a) % 3 == 1:
                    lm[:, i] += ty

    # --- global spatial scale ---
    if rng.random() < cfg.scale_p:
        s = float(rng.uniform(*cfg.scale_range))
        lm[:, xy] *= s
        gm[:, 0:126] *= s        # hand-centric coords
        gm[:, 126:166] *= s      # bone lengths

    # --- temporal speed / resample ---
    if rng.random() < cfg.time_warp_p:
        r = float(rng.uniform(*cfg.time_warp_range))
        lm = _time_warp(lm, r, order_nn=False)
        gm = _time_warp(gm, r, order_nn=False)
        av = _time_warp(av, r, order_nn=True)

    # --- coordinate jitter ---
    if cfg.coord_jitter_std > 0:
        lm += rng.normal(0.0, cfg.coord_jitter_std, lm.shape).astype(np.float32)
        gm += rng.normal(0.0, cfg.coord_jitter_std, gm.shape).astype(np.float32)

    # --- velocity / acceleration perturbation ---
    if rng.random() < cfg.motion_perturb_p:
        ms = float(rng.uniform(*cfg.motion_scale_range))
        gm[:, 217:469] = gm[:, 217:469] * ms + rng.normal(
            0.0, cfg.motion_noise_std, gm[:, 217:469].shape
        ).astype(np.float32)

    # --- frame (token) dropout ---
    if cfg.frame_dropout_p > 0:
        drop = rng.random(NUM_FRAMES) < cfg.frame_dropout_p
        if drop.all():
            drop[rng.integers(0, NUM_FRAMES)] = False
        lm[drop] = 0.0
        gm[drop] = 0.0
        av[drop] = 0.0

    # --- landmark-column dropout ---
    if rng.random() < cfg.col_dropout_p:
        k = max(1, int(534 * cfg.col_dropout_frac))
        cols = rng.choice(534, size=k, replace=False)
        lm[:, cols] = 0.0

    # --- modality dropout (never both hands) ---
    if rng.random() < cfg.modality_dropout_p:
        choice = rng.choice(["left", "right", "pose", "face"], p=[0.12, 0.12, 0.16, 0.60])
        if choice == "left":
            lm[:, 0:63] = 0.0
            gm[:, 0:63] = 0.0
            gm[:, 126:146] = 0.0
            gm[:, 166:181] = 0.0
            gm[:, 217:280] = 0.0
            gm[:, 343:406] = 0.0
            av[:, 0] = 0.0
        elif choice == "right":
            lm[:, 63:126] = 0.0
            gm[:, 63:126] = 0.0
            gm[:, 146:166] = 0.0
            gm[:, 181:196] = 0.0
            gm[:, 280:343] = 0.0
            gm[:, 406:469] = 0.0
            av[:, 1] = 0.0
        elif choice == "pose":
            lm[:, 126:258] = 0.0
            av[:, 2] = 0.0
        else:
            lm[:, 258:534] = 0.0
            av[:, 3] = 0.0

    return (
        np.ascontiguousarray(lm, dtype=np.float32),
        np.ascontiguousarray(gm, dtype=np.float32),
        np.ascontiguousarray(av, dtype=np.float32),
    )
