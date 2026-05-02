"""B6: WhisperX alignment refines word boundaries. Plan §B6.

The alignment model is ~3 GB and cached at module level (Plan B6 Refactor).
Tests are skipped when whisperx isn't installed — operator captures the
golden once on a GPU box and pins the result under tests/fixtures/golden/.
"""

from typing import List, Mapping

_MODEL_CACHE = {}


def align_with_whisperx(segments, audio_path: str, language: str = "en"):
    """Align segments by re-anchoring word boundaries to forced phoneme
    positions. Preserves word strings (set + order); adjusts start/end.

    Returns: list of segments in the same shape, with refined `words[]`.
    """
    import whisperx
    import torch

    cache_key = (language, "cuda" if torch.cuda.is_available() else "cpu")
    if cache_key not in _MODEL_CACHE:
        device = cache_key[1]
        _MODEL_CACHE[cache_key] = whisperx.load_align_model(
            language_code=language, device=device
        )
    model_a, metadata = _MODEL_CACHE[cache_key]
    audio = whisperx.load_audio(audio_path)
    device = cache_key[1]
    return whisperx.align(
        segments, model_a, metadata, audio, device,
        return_char_alignments=False,
    )["segments"]


def f1_at_tolerance(
    predicted_ts: List[float],
    expected_ts: List[float],
    tolerance_s: float = 0.05,
) -> float:
    """Binary F1 of boundary placement.

    A prediction is a true positive iff it lands within ±tolerance_s of
    the expected boundary. Plan §B6 + Review C3: research §13 Q3 cites
    F1≈0.79 @ 50ms as the floor — this is the canonical metric, not the
    weaker `mean_delta` proxy that the original plan used.
    """
    tp = sum(
        1 for p, e in zip(predicted_ts, expected_ts) if abs(p - e) <= tolerance_s
    )
    precision = tp / max(len(predicted_ts), 1)
    recall = tp / max(len(expected_ts), 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)
