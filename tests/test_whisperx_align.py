"""B6: WhisperX alignment + F1@50ms boundary precision.

Two layers:
- f1_at_tolerance: PURE function, fully tested without whisperx installed.
- align_with_whisperx: integration; skipped when whisperx unavailable OR
  when the golden fixture isn't present (operator captures it once).
"""

import importlib.util
import json
from pathlib import Path

import pytest

from align_with_whisperx import f1_at_tolerance

HAS_WHISPERX = importlib.util.find_spec("whisperx") is not None
GOLDEN = Path(__file__).parent / "fixtures" / "golden" / "10s_speech__words_aligned_golden.json"


# ─── f1_at_tolerance: pure function tests ────────────────────────────────────


def test_f1_perfect_score_when_all_predictions_exact():
    predicted = [1.0, 2.0, 3.0]
    expected = [1.0, 2.0, 3.0]
    assert f1_at_tolerance(predicted, expected, tolerance_s=0.05) == pytest.approx(1.0)


def test_f1_perfect_score_within_tolerance_window():
    predicted = [1.04, 1.99, 3.03]  # all within ±50ms
    expected = [1.00, 2.00, 3.00]
    assert f1_at_tolerance(predicted, expected, tolerance_s=0.05) == pytest.approx(1.0)


def test_f1_zero_when_all_predictions_outside_tolerance():
    predicted = [10.0, 20.0, 30.0]
    expected = [1.0, 2.0, 3.0]
    score = f1_at_tolerance(predicted, expected, tolerance_s=0.05)
    assert score == pytest.approx(0.0)


def test_f1_partial_credit_for_partial_hits():
    # 2 of 4 within tolerance → precision=recall=0.5 → F1=0.5
    predicted = [1.0, 2.0, 99.0, 99.0]
    expected = [1.0, 2.0, 3.0, 4.0]
    score = f1_at_tolerance(predicted, expected, tolerance_s=0.05)
    assert score == pytest.approx(0.5)


def test_f1_handles_empty_inputs_without_division_error():
    assert f1_at_tolerance([], [], tolerance_s=0.05) == pytest.approx(0.0)


# ─── align_with_whisperx: integration (skipped without whisperx) ─────────────


@pytest.mark.skipif(not HAS_WHISPERX, reason="whisperx not installed")
@pytest.mark.skipif(not GOLDEN.exists(), reason="golden fixture not yet captured")
def test_alignment_preserves_words_and_clears_f1_floor():
    """Run alignment on the 10s fixture and verify F1@50ms ≥ 0.79."""
    from align_with_whisperx import align_with_whisperx

    audio_fixture = Path(__file__).parent / "fixtures" / "10s_speech.wav"
    raw_path = Path(__file__).parent / "fixtures" / "10s_speech__words_raw.json"
    raw = json.loads(raw_path.read_text())
    aligned = align_with_whisperx(raw, str(audio_fixture), language="en")
    raw_words = [w["word"] for s in raw for w in s["words"]]
    aligned_words = [w["word"] for s in aligned for w in s["words"]]
    assert raw_words == aligned_words
    expected = json.loads(GOLDEN.read_text())
    pred_ends = [w["end"] for s in aligned for w in s["words"]]
    gold_ends = [w["end"] for s in expected for w in s["words"]]
    score = f1_at_tolerance(pred_ends, gold_ends, tolerance_s=0.05)
    assert score >= 0.79, f"F1 @ 50ms = {score:.3f} below research floor 0.79"
