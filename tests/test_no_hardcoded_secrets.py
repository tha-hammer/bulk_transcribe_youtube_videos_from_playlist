"""B1: no hardcoded API key + module fails fast when env unset.

Plan §B1: vendor/bulk_transcribe_youtube_videos_from_playlist must read
OPENAI_API_KEY from the environment. Hardcoded `sk-…` strings forbidden;
empty/missing env raises at module load when USE_OPENAI_API=1.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parent.parent / "bulk_transcribe_youtube_videos_from_playlist.py"
MODULE_DIR = SOURCE.parent


def test_no_hardcoded_openai_key_in_source():
    contents = SOURCE.read_text()
    assert not re.search(
        r"openai_api_key\s*=\s*['\"]sk-",
        contents,
    ), "Hardcoded OpenAI API key (sk-...) found in source — must read from env"


def test_no_replace_with_your_api_key_placeholder_either():
    """The upstream placeholder is also a footgun — easy to commit by accident."""
    contents = SOURCE.read_text()
    assert "REPLACE_WITH_YOUR_API_KEY" not in contents, (
        "Placeholder string still in source — should be replaced with env-var read"
    )


def test_module_fails_fast_when_env_unset_but_use_openai_set():
    """USE_OPENAI_API=1 with no OPENAI_API_KEY -> KeyError at module load."""
    env = {
        "USE_OPENAI_API": "1",
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    # Strip any inherited OPENAI_API_KEY so the test is deterministic.
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-c", "import bulk_transcribe_youtube_videos_from_playlist"],
        cwd=str(MODULE_DIR),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0, "Module should have failed at load when OPENAI_API_KEY missing"
    assert "OPENAI_API_KEY" in result.stderr or "KeyError" in result.stderr, (
        f"Expected KeyError mentioning OPENAI_API_KEY in stderr; got:\n{result.stderr}"
    )


def test_module_raises_valueerror_when_key_set_but_empty():
    """Empty string is treated as missing — better than silently authenticating with ''."""
    env = {
        "USE_OPENAI_API": "1",
        "OPENAI_API_KEY": "   ",  # whitespace-only
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import bulk_transcribe_youtube_videos_from_playlist"],
        cwd=str(MODULE_DIR),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "OPENAI_API_KEY" in result.stderr and "empty" in result.stderr.lower()


def test_module_loads_cleanly_when_use_openai_unset():
    """Local-whisper path: USE_OPENAI_API=0 (or unset) — no env-key required."""
    env = {
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    env.pop("USE_OPENAI_API", None)
    env.pop("OPENAI_API_KEY", None)
    # Module load may still fail on missing optional imports (psutil etc.),
    # but it must NOT fail with KeyError on OPENAI_API_KEY.
    result = subprocess.run(
        [sys.executable, "-c", "import bulk_transcribe_youtube_videos_from_playlist"],
        cwd=str(MODULE_DIR),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        # Tolerate ImportError on optional deps; reject any KeyError on OPENAI_API_KEY.
        assert "OPENAI_API_KEY" not in result.stderr, (
            f"Module raised on OPENAI_API_KEY despite USE_OPENAI_API unset:\n{result.stderr}"
        )
