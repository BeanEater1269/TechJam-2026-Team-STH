"""
Shared inference module for the webdemo -- used by both /predict (single image,
index.html) and /api/live/analyze (batch, live.html). One implementation, so the two
surfaces can never disagree on a score.

Production model is fixed: FiLMClassifier + CLIP ViT-L/14 (768-dim). No other
model/backbone is served -- b32 embeddings were retired, results_b32/ is a frozen
historical snapshot only.

Preprocessing MUST match scripts/data_prep/build_cache.py's make_clean_baseline() (the
real-image branch: crop_to_square() if non-square, then resize_to_working() to 512x512)
-- that's what every training/eval image went through before CLIP embeddings and the 4
signals were computed on it. Skipping this, or doing it differently, feeds the model
data far outside what it was trained on: FiLM uses the signals to modulate the CLIP
embedding directly, so bad signals corrupt the prediction itself, not just a side
channel. jitter_crop_square() is intentionally NOT used here -- it's a fakes-only
anti-fingerprint step from build_cache.py, and we don't know the true label at
inference time, so every uploaded image goes through the same real-image path.

transformers>=5 changed CLIPModel.get_image_features() to return a
BaseModelOutputWithPooling instead of a bare tensor -- `.pooler_output` is the correct
768-dim accessor (confirmed: bit-identical to the classic projected embedding). Do not
copy extract_embeddings.py's `.get_image_features(**inputs).cpu().numpy()` verbatim;
that line is currently broken on this transformers version.
"""
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "features"))

from model_film import FiLMClassifier  # noqa: E402
from normalize import apply_normalization, compute_train_stats  # noqa: E402
from signals import dct_band_energies, laplacian_variance, noise_variance  # noqa: E402
from transforms import WORKING_RES, crop_to_square, resize_to_working  # noqa: E402

SIGNAL_COLUMNS = ["laplacian_var", "dct_low_energy", "dct_high_energy", "noise_variance"]
CLIP_CHECKPOINT = "openai/clip-vit-large-patch14"
BACKBONE_LABEL, BACKBONE_DIM = "ViT-L/14", 768
FILM_CHECKPOINT = REPO_ROOT / "checkpoints" / "model_film_normalized.pt"
TRAIN_STATS_PATH = REPO_ROOT / "data" / "cache" / "stats" / "train_signals.npz"
THRESHOLD = 0.5
CLIP_BATCH_SIZE = 16  # extract_embeddings.py uses 64 offline; smaller here since this
                      # runs inside a web request, not a batch job with the machine to itself


def standardize(image: Image.Image) -> Image.Image:
    """Same shape/resolution standardization build_cache.py applies to real images --
    phone photos aren't auto-rotated by PIL (training data had no such rotation, so an
    un-transposed upload is a real live-only failure mode), then crop-to-square (never
    squash), then resize to WORKING_RES."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if w != h:
        image = crop_to_square(image)
        w = h = min(w, h)
    # Skip a redundant resize when already exactly WORKING_RES -- e.g. when re-running
    # inference on a previously-saved standardized image (results_judge/.../images/).
    # PIL's BICUBIC resize isn't a perfect identity at equal size, so skipping it here
    # (scoped to this module only, not build_cache.py's shared resize_to_working) keeps
    # that round-trip exactly reproducible instead of drifting by a fraction of a percent.
    if (w, h) != (WORKING_RES, WORKING_RES):
        image = resize_to_working(image)
    return image


def compute_signals(std_image: Image.Image) -> np.ndarray:
    """Raw (unnormalized) 4 signals for one already-standardized image, in
    SIGNAL_COLUMNS order -- mirrors signals.py's compute_stats_for_path(), but works
    in-memory since an uploaded image (single-image path) never touches disk."""
    gray = np.asarray(std_image.convert("L"), dtype=np.float64)
    lap_var = laplacian_variance(gray)
    dct_low, dct_high = dct_band_energies(gray)
    noise_var = noise_variance(gray)
    return np.array([lap_var, dct_low, dct_high, noise_var], dtype=np.float32)


class Predictor:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model = CLIPModel.from_pretrained(CLIP_CHECKPOINT).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_CHECKPOINT)

        self.film = FiLMClassifier(clip_dim=BACKBONE_DIM, signal_dim=len(SIGNAL_COLUMNS))
        self.film.load_state_dict(torch.load(FILM_CHECKPOINT, map_location=self.device))
        self.film.to(self.device).eval()

        self.signal_mean, self.signal_std = compute_train_stats(TRAIN_STATS_PATH, SIGNAL_COLUMNS)

        self._lock = threading.Lock()

    def _embed(self, images: list) -> np.ndarray:
        chunks = []
        for i in range(0, len(images), CLIP_BATCH_SIZE):
            batch = images[i : i + CLIP_BATCH_SIZE]
            inputs = self.clip_processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.clip_model.get_image_features(**inputs)
            chunks.append(out.pooler_output.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def predict(self, images: list) -> list:
        """images: list of PIL.Image (any size/mode). Returns one dict per image,
        same order as input:
          raw_prob    -- sigmoid output, P(fake). THE inference result.
          label       -- "FAKE" if raw_prob >= THRESHOLD else "REAL"
          confidence  -- confidence in the ASSIGNED label: raw_prob if FAKE else
                          1-raw_prob. Always in [0.5, 1.0] -- distinct from raw_prob,
                          which is a fixed "P(fake)" regardless of which label won.
          signals_raw -- dict of the 4 raw (unnormalized) signal values
          std_image   -- the standardized PIL.Image actually fed to the model (caller's
                          responsibility to persist or discard; not JSON-serializable)
        """
        with self._lock:
            std_images = [standardize(im) for im in images]
            embeddings = self._embed(std_images)
            raw_signals = np.stack([compute_signals(im) for im in std_images], axis=0)
            norm_signals = apply_normalization(raw_signals, self.signal_mean, self.signal_std)

            emb_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            sig_t = torch.tensor(norm_signals, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.film(emb_t, sig_t)).cpu().numpy()

        results = []
        for i, p in enumerate(probs):
            raw_prob = float(p)
            label = "FAKE" if raw_prob >= THRESHOLD else "REAL"
            confidence = raw_prob if label == "FAKE" else 1.0 - raw_prob
            results.append({
                "raw_prob": raw_prob,
                "label": label,
                "confidence": confidence,
                "signals_raw": dict(zip(SIGNAL_COLUMNS, raw_signals[i].tolist())),
                "std_image": std_images[i],
            })
        return results


_PREDICTOR: Predictor | None = None
_PREDICTOR_LOCK = threading.Lock()


def get_predictor() -> Predictor:
    """Lazy singleton -- double-checked locking so concurrent first requests don't each
    load their own CLIP+FiLM copy."""
    global _PREDICTOR
    if _PREDICTOR is None:
        with _PREDICTOR_LOCK:
            if _PREDICTOR is None:
                _PREDICTOR = Predictor()
    return _PREDICTOR


def to_record(result: dict) -> dict:
    """Strips the non-JSON-serializable std_image out of a predict() result dict."""
    return {k: v for k, v in result.items() if k != "std_image"}
