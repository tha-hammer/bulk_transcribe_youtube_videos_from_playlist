"""Test-time stubs for heavy ML/audio imports.

The transcriber module pulls in openai/faster_whisper/pytubefix/pandas/
numba/pydub/tqdm at import time. Tests for B3/B4/B7 only exercise pure
helpers (`_serialize_words`, `vad_snap`) that do NOT touch those deps,
so we install lightweight stubs in sys.modules BEFORE any test imports
the module.

This file is conftest.py at the package root for pytest auto-discovery.
"""

import sys
import types


def _stub(name: str, **attrs):
    """Create a fake module exposing the given attributes."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Top-level stubs — only those the transcriber imports unconditionally.
_stub("psutil", cpu_count=lambda **_kw: 4)
_stub("openai", AsyncOpenAI=type("AsyncOpenAI", (), {}))
_stub("pytubefix",
      YouTube=type("YouTube", (), {}),
      Playlist=type("Playlist", (), {}))
_stub("pandas")
_stub("faster_whisper", WhisperModel=type("WhisperModel", (), {}))
_stub("numba", cuda=type("Cuda", (), {"is_available": staticmethod(lambda: False)})())
_stub("pydub", AudioSegment=type("AudioSegment", (), {}))
_stub("tqdm", tqdm=lambda *a, **kw: types.SimpleNamespace(
    update=lambda *a, **kw: None, __enter__=lambda self: self, __exit__=lambda *a: None))

# spacy: nlp = spacy.load(...) at module load. Stub so the call returns a
# fake nlp object with a callable interface returning a doc with `.sents`.
def _fake_nlp_load(_model):
    fake_doc = lambda _text: types.SimpleNamespace(
        sents=iter([types.SimpleNamespace(text="stub sentence")])
    )
    return fake_doc

_spacy = _stub("spacy", load=_fake_nlp_load)
_stub("spacy.cli", download=lambda *a, **kw: None)
_spacy.cli = sys.modules["spacy.cli"]  # mirror submodule on parent
