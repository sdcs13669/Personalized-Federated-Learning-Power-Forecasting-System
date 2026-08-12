"""Centralised training / evaluation parameters — single source of truth.

All training and visualisation scripts read their defaults from here
(``train_baseline``, ``train_personalized``, ``train_dp_personalized``,
``visualize_eval``, ``data_utils``, ``train_eval_utils``, ``models/config``).
Tune once, apply everywhere.

Invariants
----------
- ``STRIDE == PRED_LEN`` → rolling evaluation is gap-free (every timestep
  is predicted exactly once) and training windows slide by the output
  window, which suits the real-time scheduling setting (predict 3 h ahead,
  re-forecast every 3 h).
- Window geometry must stay consistent with ``models/config.yaml``.
"""

# ---------------------------------------------------------------------------
# Window geometry (real-time scheduling: 3 days in → 3 hours out)
# ---------------------------------------------------------------------------
INPUT_STEPS = 144        # input window: 3 days @ 30-min steps
PRED_LEN = 6             # output window: 3 hours @ 30-min steps
STRIDE = PRED_LEN        # sliding step = output window → continuous coverage
TRAIN_RATIO = 0.8        # chronological split (μ/σ statistics + eval split)

# ---------------------------------------------------------------------------
# Phase 2 — FedAvg baseline (Flower)
# ---------------------------------------------------------------------------
BASELINE_ROUNDS = 20
BASELINE_LOCAL_EPOCHS = 1
BASELINE_LR = 1e-3
BASELINE_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Phase 3/4 — per-client Corrector training
# ---------------------------------------------------------------------------
CORRECTOR_EPOCHS = 30
CORRECTOR_LR = 1e-3
CORRECTOR_BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Phase 4 — DP-SGD defaults (Opacus)
# ---------------------------------------------------------------------------
DP_NOISE_MULTIPLIERS = (0.5, 1.0, 2.0, 5.0)
DP_MAX_GRAD_NORM = 1.0
DP_DELTA = 1e-5

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
DISPLAY_STEPS = 7 * 48   # 7 days @ 30-min steps
