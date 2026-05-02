"""B7: vad_snap unit tests with injected silence boundaries (no silero_vad
required at test time)."""

import pytest

from vad_snap import vad_snap


def fake_boundaries(boundaries):
    """Return a fake boundaries_fn that always emits the given timestamps."""
    return lambda _path: list(boundaries)


def test_vad_snap_pulls_word_end_to_silence_within_tolerance():
    words = [
        {"word": "hello", "start": 0.10, "end": 0.42, "probability": 0.95},
        {"word": "world", "start": 0.55, "end": 0.95, "probability": 0.93},
    ]
    snapped = vad_snap(
        words,
        "/dev/null",
        tolerance_s=0.2,
        boundaries_fn=fake_boundaries([0.50, 1.10]),
    )
    # 0.42 → 0.50 (delta 0.08 within 0.2 tolerance)
    assert snapped[0]["end"] == pytest.approx(0.50)
    # 0.95: nearest is 1.10 (delta 0.15 within 0.2 tolerance) — snaps
    assert snapped[1]["end"] == pytest.approx(1.10)


def test_vad_snap_leaves_word_alone_when_no_silence_within_tolerance():
    words = [{"word": "in", "start": 5.0, "end": 5.2, "probability": 0.9}]
    snapped = vad_snap(
        words,
        "/dev/null",
        tolerance_s=0.05,
        boundaries_fn=fake_boundaries([7.0, 8.0]),
    )
    assert snapped[0]["end"] == 5.2  # unchanged


def test_vad_snap_preserves_other_fields_unchanged():
    words = [{"word": "a", "start": 0.1, "end": 0.4, "probability": 0.91}]
    snapped = vad_snap(
        words,
        "/dev/null",
        tolerance_s=0.2,
        boundaries_fn=fake_boundaries([0.5]),
    )
    assert snapped[0]["word"] == "a"
    assert snapped[0]["start"] == 0.1
    assert snapped[0]["probability"] == 0.91


def test_vad_snap_tiebreak_prefers_earlier_candidate():
    """Two boundaries equidistant from word.end → earlier one wins."""
    words = [{"word": "x", "start": 0.0, "end": 0.50, "probability": 0.9}]
    snapped = vad_snap(
        words,
        "/dev/null",
        tolerance_s=0.2,
        boundaries_fn=fake_boundaries([0.40, 0.60]),
    )
    assert snapped[0]["end"] == pytest.approx(0.40)


def test_vad_snap_only_touches_word_end_not_start():
    """Review W8: snap targets `end` only; `start` is the alignment's job."""
    words = [{"word": "x", "start": 0.45, "end": 0.50, "probability": 0.9}]
    snapped = vad_snap(
        words,
        "/dev/null",
        tolerance_s=0.2,
        boundaries_fn=fake_boundaries([0.10]),  # would pull start if we did
    )
    assert snapped[0]["start"] == 0.45  # unchanged


def test_vad_snap_returns_new_list_does_not_mutate_input():
    words = [{"word": "x", "start": 0.0, "end": 0.4, "probability": 0.9}]
    original = dict(words[0])
    vad_snap(words, "/dev/null", tolerance_s=0.2, boundaries_fn=fake_boundaries([0.5]))
    assert words[0] == original
