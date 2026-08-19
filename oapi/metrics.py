from __future__ import annotations
from typing import Optional, Dict, Any
import math
import numpy as np


def success_probability(best_energy: np.ndarray, target_energy: float, atol: float = 1e-9) -> float:
    return float(np.mean(np.asarray(best_energy) <= target_energy + atol))


def tts(run_time: float, p_success: float, confidence: float = 0.99) -> float:
    """TTS_p = T_run log(1-p)/log(1-p_succ)."""
    if p_success <= 0:
        return float("inf")
    if p_success >= 1:
        return float(run_time)
    return float(run_time * math.log(1.0 - confidence) / math.log(1.0 - p_success))


def summarize_result(result, target_energy: Optional[float] = None, confidence: float = 0.99) -> Dict[str, Any]:
    out = dict(result.summary)
    if target_energy is not None:
        ps = success_probability(result.best_energy, target_energy)
        out["target_energy"] = float(target_energy)
        out["p_success"] = ps
        out[f"tts_{confidence:.2f}"] = tts(result.runtime_s, ps, confidence)
    return out


def normalized_gap(value: float, reference: float) -> float:
    denom = max(abs(reference), 1e-12)
    return float((value - reference) / denom)
