"""B3 + B4: per-word timestamps + monotonicity invariant.

Unit-level: exercises `_serialize_words` against fake segment objects so
the test runs without `faster_whisper` (or any ML deps) installed. The
ACTUAL transcribe-the-audio integration test is operator-managed (run on
a GPU box with the model downloaded).
"""

from dataclasses import dataclass

import bulk_transcribe_youtube_videos_from_playlist as mod


@dataclass
class FakeWord:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class FakeSegment:
    words: list


def test_serialize_words_emits_required_fields():
    seg = FakeSegment(words=[
        FakeWord("hello", 0.10, 0.42, 0.951),
        FakeWord("world", 0.55, 0.95, 0.934),
    ])
    out = mod._serialize_words(seg)
    assert len(out) == 2
    for w in out:
        assert set(w.keys()) == {"word", "start", "end", "probability"}
    assert out[0]["word"] == "hello"
    assert out[0]["start"] == 0.10
    assert out[0]["end"] == 0.42
    assert out[0]["probability"] == 0.951


def test_serialize_words_handles_segment_without_words_attr():
    """Some engines may return segments without a `words` attribute. Per
    B3 contract, those serialize as an empty list (graceful)."""
    class NoWordsAttr:
        pass

    out = mod._serialize_words(NoWordsAttr())
    assert out == []


def test_serialize_words_handles_segment_words_is_none():
    seg = FakeSegment(words=None)
    out = mod._serialize_words(seg)
    assert out == []


def test_serialize_words_rounds_to_three_decimals():
    seg = FakeSegment(words=[FakeWord("x", 0.1234567, 0.7654321, 0.9876543)])
    out = mod._serialize_words(seg)
    assert out[0]["start"] == 0.123
    assert out[0]["end"] == 0.765
    assert out[0]["probability"] == 0.988


def test_b4_words_monotonic_in_source_order():
    """B4 invariant: serialize_words preserves source order; subsequent
    words have non-decreasing start times. Transcribers that violate this
    must surface here (a sort step would land in `_serialize_words`)."""
    seg = FakeSegment(words=[
        FakeWord("a", 0.10, 0.20, 0.95),
        FakeWord("b", 0.25, 0.40, 0.94),
        FakeWord("c", 0.45, 0.60, 0.93),
    ])
    out = mod._serialize_words(seg)
    for prev, nxt in zip(out, out[1:]):
        assert prev["start"] <= prev["end"], f"word.start > word.end in {prev}"
        assert prev["end"] <= nxt["start"] + 0.05, (
            f"non-monotonic boundary {prev} → {nxt}"
        )


def test_source_has_word_timestamps_true_in_transcribe_call():
    """B3 source assertion: the model.transcribe call must pass
    word_timestamps=True so per-word timing actually flows."""
    src = (mod.__file__ and open(mod.__file__).read()) or ""
    assert "word_timestamps=True" in src, (
        "model.transcribe(...) must be called with word_timestamps=True"
    )
