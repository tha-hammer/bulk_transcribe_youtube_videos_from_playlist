"""B7: snap WhisperX-aligned word ENDS to the nearest detected silence
boundary within a tolerance window. Plan §B7. Review W8: only word.end
is snapped (word.start is left to alignment).

Design note: the silence-boundary detector is INJECTABLE so tests run
without silero_vad installed. Production code uses
`default_silence_boundaries` which loads silero_vad lazily.
"""

from typing import Callable, List, Mapping, Optional


def vad_snap(
    words: List[Mapping[str, float]],
    audio_path: str,
    tolerance_s: float = 0.2,
    boundaries_fn: Optional[Callable[[str], List[float]]] = None,
) -> List[dict]:
    """Snap each word.end to the nearest silence boundary within ±tolerance_s.

    Args:
        words: list of {"word", "start", "end", "probability"} dicts
        audio_path: path to the source audio file (passed to boundaries_fn)
        tolerance_s: maximum delta to snap (default 200ms)
        boundaries_fn: returns silence boundary timestamps (seconds). When
            None, uses silero_vad via `default_silence_boundaries`. Tests
            inject a fake.

    Returns: a NEW list of word dicts with `end` adjusted; non-end fields
    are preserved. If no silence is within tolerance for a word, its `end`
    is unchanged. Deterministic tiebreak: prefer the EARLIER of two
    equidistant candidates (per Plan §B7 amendment).
    """
    boundaries = (boundaries_fn or default_silence_boundaries)(audio_path)
    out: List[dict] = []
    for w in words:
        end = w["end"]
        nearest = end
        best_delta = tolerance_s + 1e-9  # strictly larger than tolerance
        for s in boundaries:
            delta = abs(s - end)
            if delta > tolerance_s:
                continue
            # Tiebreak: prefer EARLIER candidate when deltas are equal.
            if delta < best_delta or (delta == best_delta and s < nearest):
                nearest = s
                best_delta = delta
        out.append({**w, "end": nearest})
    return out


def default_silence_boundaries(audio_path: str, sample_rate: int = 16000) -> List[float]:
    """Lazy-loaded silero_vad path. Imports happen inside the fn so the
    module loads cleanly without silero_vad installed (production runner
    needs it; unit tests use the injection point)."""
    from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

    model = load_silero_vad()
    audio = read_audio(audio_path, sampling_rate=sample_rate)
    speech_ts = get_speech_timestamps(audio, model, sampling_rate=sample_rate)
    # End of a SPEECH segment == start of the following silence — we want
    # those points so word.end can snap to "where speech stopped."
    return [s["end"] / sample_rate for s in speech_ts]
