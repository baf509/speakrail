#!/usr/bin/env python3
"""origin_allowed: a browser page may open a session only from this server's own origin, or one named in ALLOWED_ORIGINS.

    python -m pytest tests/test_origins.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataclasses import replace  # noqa: E402

from speakrail.gateway import live  # noqa: E402
from speakrail.gateway.live import origin_allowed  # noqa: E402


def req(origin, host="127.0.0.1:18114"):
    return SimpleNamespace(headers={} if origin is None else {"Origin": origin}, host=host)


def with_allowed(value, fn):
    old = live.CFG                                     # CFG is frozen: swap the module's copy
    live.CFG = replace(old, allowed_origins=value)
    try:
        return fn()
    finally:
        live.CFG = old


def test_non_browser_and_own_origin():
    assert with_allowed("", lambda: origin_allowed(req(None)))
    assert with_allowed("", lambda: origin_allowed(req("http://127.0.0.1:18114")))


def test_foreign_origin_refused_by_default():
    assert not with_allowed("", lambda: origin_allowed(req("http://localhost:3000")))
    assert not with_allowed("", lambda: origin_allowed(req("https://evil.example")))


def test_named_origins_allowed_exactly():
    allowed = "http://localhost:3000, http://127.0.0.1:3000/"
    assert with_allowed(allowed, lambda: origin_allowed(req("http://localhost:3000")))
    assert with_allowed(allowed, lambda: origin_allowed(req("http://127.0.0.1:3000")))
    assert not with_allowed(allowed, lambda: origin_allowed(req("http://localhost:3001")))
    assert not with_allowed(allowed, lambda: origin_allowed(req("https://localhost:3000")))
    assert not with_allowed(allowed, lambda: origin_allowed(req("http://localhost:3000.evil.example")))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
