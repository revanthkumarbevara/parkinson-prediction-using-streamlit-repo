"""
feature_extractor.py
Extracts the 22 vocal biomarkers required by the model from a WAV recording.

All features mirror what is in the UCI Parkinson's dataset:
  https://archive.ics.uci.edu/ml/datasets/parkinsons

Dependencies: librosa, praat-parselmouth, numpy, scipy
"""

import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run Praat via parselmouth
# ─────────────────────────────────────────────────────────────────────────────

def _pitch_object(snd, f0_min=75, f0_max=600):
    return call(snd, "To Pitch", 0.0, f0_min, f0_max)


def _point_process(snd, pitch, f0_min=75, f0_max=600):
    return call([snd, pitch], "To PointProcess (cc)")


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental frequency features (Hz)
# ─────────────────────────────────────────────────────────────────────────────

def _fo_features(pitch):
    """Returns Fo_mean, Fo_max, Fo_min in Hz."""
    f0_values = [
        call(pitch, "Get value at time", t, "Hertz", "Linear")
        for t in np.linspace(pitch.xmin, pitch.xmax, 300)
    ]
    f0_values = [v for v in f0_values if str(v) != "nan" and v > 0]
    if not f0_values:
        raise ValueError("No voiced frames detected — check your recording.")
    return float(np.mean(f0_values)), float(np.max(f0_values)), float(np.min(f0_values))


# ─────────────────────────────────────────────────────────────────────────────
# Jitter features (period irregularity)
# ─────────────────────────────────────────────────────────────────────────────

def _jitter_features(snd, pp):
    jitter_local     = call(pp, "Get jitter (local)",     0, 0, 0.0001, 0.02, 1.3)
    jitter_abs       = call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap       = call(pp, "Get jitter (rap)",       0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5      = call(pp, "Get jitter (ppq5)",      0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp       = call(pp, "Get jitter (ddp)",       0, 0, 0.0001, 0.02, 1.3)
    return jitter_local, jitter_abs, jitter_rap, jitter_ppq5, jitter_ddp


# ─────────────────────────────────────────────────────────────────────────────
# Shimmer features (amplitude irregularity)
# ─────────────────────────────────────────────────────────────────────────────

def _shimmer_features(snd, pp):
    shimmer_local    = call([snd, pp], "Get shimmer (local)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db       = call([snd, pp], "Get shimmer (local_dB)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3     = call([snd, pp], "Get shimmer (apq3)",      0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5     = call([snd, pp], "Get shimmer (apq5)",      0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11    = call([snd, pp], "Get shimmer (apq11)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda      = call([snd, pp], "Get shimmer (dda)",       0, 0, 0.0001, 0.02, 1.3, 1.6)
    return shimmer_local, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_apq11, shimmer_dda


# ─────────────────────────────────────────────────────────────────────────────
# Noise ratios
# ─────────────────────────────────────────────────────────────────────────────

def _noise_features(snd):
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    # NHR = 1/HNR approximation (dataset uses linear ratio; use amplitude-based)
    nhr = 1.0 / hnr if hnr > 0 else 0.0
    return float(nhr), float(hnr)


# ─────────────────────────────────────────────────────────────────────────────
# Nonlinear dynamical features (via librosa)
# ─────────────────────────────────────────────────────────────────────────────

def _nonlinear_features(audio: np.ndarray, sr: int):
    # ── Ensure audio is strictly 1-D float32 ─────────────────────────────────
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    audio = audio.astype(np.float32)

    # ── RPDE: spectral entropy proxy ──────────────────────────────────────────
    S = np.abs(librosa.stft(audio))                          # (freq, time)
    power = S ** 2
    col_sums = power.sum(axis=0, keepdims=True) + 1e-10
    power_norm = power / col_sums
    spectral_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10), axis=0)
    rpde = float(np.mean(spectral_entropy) / np.log2(S.shape[0] + 1))

    # ── DFA: detrended fluctuation slope ─────────────────────────────────────
    rms = librosa.feature.rms(y=audio)[0]                    # 1-D
    n_max = min(len(rms) // 2, 64)
    n_vals = np.arange(2, max(n_max, 3))
    if len(n_vals) < 4:
        dfa = 0.6
    else:
        flucts = []
        for n in n_vals:
            trim = len(rms) - len(rms) % n
            if trim == 0:
                continue
            seg = rms[:trim].reshape(-1, n)                  # (num_segs, n)
            x   = np.arange(n, dtype=float)
            # fit each segment independently — avoids (2,) vs (22,) broadcast
            detrended = []
            for s in seg:
                coeffs = np.polyfit(x, s, 1)
                trend  = np.polyval(coeffs, x)
                detrended.append(np.mean((s - trend) ** 2))
            flucts.append(np.sqrt(np.mean(detrended)))
        if len(flucts) < 2:
            dfa = 0.6
        else:
            log_n = np.log(n_vals[:len(flucts)].astype(float))
            log_f = np.log(np.array(flucts) + 1e-10)
            dfa   = float(np.polyfit(log_n, log_f, 1)[0])

    # ── D2: correlation dimension proxy ──────────────────────────────────────
    mfccs   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # (13, time)
    cov     = np.cov(mfccs)                                     # (13, 13)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    d2      = float(abs(
        np.sum(eigvals * np.log(eigvals)) / (np.log(np.sum(eigvals)) + 1e-10)
    ))

    # ── Pitch period entropy & spread ────────────────────────────────────────
    f0, _, _ = librosa.pyin(audio, fmin=75.0, fmax=600.0, sr=sr)
    f0_clean  = f0[~np.isnan(f0)]
    if len(f0_clean) < 2:
        f0_clean = np.array([150.0, 155.0])

    spread1 = float(np.std(f0_clean))
    spread2 = float(np.mean(np.abs(np.diff(f0_clean))))

    hist, _ = np.histogram(f0_clean, bins=30, density=True)
    hist     = hist[hist > 0]
    ppe      = float(-np.sum(hist * np.log2(hist + 1e-10)) / np.log2(len(hist) + 1))

    return rpde, dfa, d2, ppe, spread1, spread2


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(audio_path: str) -> dict:
    """
    Extract all 22 vocal biomarkers from a WAV file.

    Returns a dict keyed by the feature names used in the training dataset.
    Raises ValueError if the audio is too short or contains no voiced frames.
    """
    # Load via librosa (resamples to mono 22050 Hz)
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    if len(audio) / sr < 1.0:
        raise ValueError("Recording is too short (< 1 second). Please record at least 3 seconds of sustained vowel 'ahh'.")

    # Parselmouth (Praat wrapper)
    snd   = parselmouth.Sound(audio_path)
    pitch = _pitch_object(snd)
    pp    = _point_process(snd, pitch)

    fo_mean, fo_max, fo_min = _fo_features(pitch)
    j_local, j_abs, j_rap, j_ppq5, j_ddp = _jitter_features(snd, pp)
    s_local, s_db, s_apq3, s_apq5, s_apq11, s_dda = _shimmer_features(snd, pp)
    nhr, hnr = _noise_features(snd)
    rpde, dfa, d2, ppe, spread1, spread2 = _nonlinear_features(audio, sr)

    return {
        "MDVP:Fo(Hz)":       fo_mean,
        "MDVP:Fhi(Hz)":      fo_max,
        "MDVP:Flo(Hz)":      fo_min,
        "MDVP:Jitter(%)":    j_local,
        "MDVP:Jitter(Abs)":  j_abs,
        "MDVP:RAP":          j_rap,
        "MDVP:PPQ":          j_ppq5,
        "Jitter:DDP":        j_ddp,
        "MDVP:Shimmer":      s_local,
        "MDVP:Shimmer(dB)":  s_db,
        "Shimmer:APQ3":      s_apq3,
        "Shimmer:APQ5":      s_apq5,
        "MDVP:APQ":          s_apq11,
        "Shimmer:DDA":       s_dda,
        "NHR":               nhr,
        "HNR":               hnr,
        "RPDE":              rpde,
        "DFA":               dfa,
        "spread1":           spread1,
        "spread2":           spread2,
        "D2":                d2,
        "PPE":               ppe,
    }