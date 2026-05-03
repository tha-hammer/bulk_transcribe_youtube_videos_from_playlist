"""B3 OpenAI path: timestamp_granularities + flat-words → per-segment buckets.

The OpenAI Whisper API returns segments AND a flat words array when
timestamp_granularities=["word","segment"] is set. _bucket_words_into_segments
reshapes them so the bridge sees the same per-segment .words[] structure
the local-whisper path produces.
"""

import bulk_transcribe_youtube_videos_from_playlist as mod


def test_source_calls_openai_with_timestamp_granularities():
    """B3 source assertion: OpenAI branch must request word timestamps so
    the bridge has timing data when running OpenAI-first (the user-stated
    primary path)."""
    src = open(mod.__file__).read()
    assert 'timestamp_granularities=["word", "segment"]' in src, (
        "OpenAI transcribe call must pass timestamp_granularities"
    )


def test_bucket_words_assigns_each_word_to_its_segment():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello world"},
        {"start": 1.0, "end": 2.0, "text": "again world"},
    ]
    flat_words = [
        {"word": "hello", "start": 0.10, "end": 0.45},
        {"word": "world", "start": 0.50, "end": 0.95},
        {"word": "again", "start": 1.05, "end": 1.45},
        {"word": "world", "start": 1.50, "end": 1.95},
    ]
    out = mod._bucket_words_into_segments(segments, flat_words)
    assert len(out) == 2
    assert [w["word"] for w in out[0]["words"]] == ["hello", "world"]
    assert [w["word"] for w in out[1]["words"]] == ["again", "world"]


def test_bucket_words_sets_probability_sentinel_to_one():
    """OpenAI doesn't expose per-word probability — sentinel 1.0 keeps the
    bridge's required-field contract intact."""
    segments = [{"start": 0.0, "end": 1.0}]
    flat_words = [{"word": "x", "start": 0.1, "end": 0.4}]
    out = mod._bucket_words_into_segments(segments, flat_words)
    assert out[0]["words"][0]["probability"] == 1.0


def test_bucket_words_handles_empty_word_list():
    segments = [{"start": 0.0, "end": 1.0}]
    out = mod._bucket_words_into_segments(segments, [])
    assert out[0]["words"] == []


def test_bucket_words_handles_words_without_matching_segment():
    """Words past the last segment's end are dropped (no-op rather than
    crash). Logging this is out of scope for the unit; integration test
    on a real cascade run will surface drops."""
    segments = [{"start": 0.0, "end": 1.0}]
    flat_words = [{"word": "x", "start": 0.5, "end": 0.9}, {"word": "y", "start": 1.5, "end": 1.9}]
    out = mod._bucket_words_into_segments(segments, flat_words)
    assert [w["word"] for w in out[0]["words"]] == ["x"]


def test_bucket_words_rounds_timestamps_to_three_decimals():
    segments = [{"start": 0.0, "end": 1.0}]
    flat_words = [{"word": "x", "start": 0.1234567, "end": 0.7654321}]
    out = mod._bucket_words_into_segments(segments, flat_words)
    assert out[0]["words"][0]["start"] == 0.123
    assert out[0]["words"][0]["end"] == 0.765
