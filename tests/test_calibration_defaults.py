"""Umbral y margen calibrados por backend: se aplican solos y no pisan lo que pide el cliente."""
from dataclasses import replace

import pytest

from odin_vision.config import BACKEND_DEFAULTS, Settings, optional_ratio
from odin_vision.engine import Engine
from odin_vision.schemas import FilterConfig


def test_session_starts_with_backend_defaults(engine):
    session = engine.create()
    assert (session.config.threshold, session.config.margin) == BACKEND_DEFAULTS["demo"]


def test_models_backend_uses_calibrated_values(settings):
    threshold, margin = replace(settings, backend="models").recognition_defaults()
    assert (threshold, margin) == BACKEND_DEFAULTS["models"]
    # El encoder aprendido agrupa por categoría: exige más similitud, pero un margen alto bloquearía casi
    # todo reconocimiento cuando hay varias instancias parecidas registradas en la misma categoría.
    assert threshold > BACKEND_DEFAULTS["demo"][0] and margin < BACKEND_DEFAULTS["demo"][1]


def test_explicit_values_are_preserved(engine):
    sid = engine.create().id
    config = engine.configure(sid, FilterConfig(detect=["object"], threshold=0.62))
    assert config["threshold"] == pytest.approx(0.62)
    assert config["margin"] == BACKEND_DEFAULTS["demo"][1]
    assert engine.session(sid).config.threshold == pytest.approx(0.62)


def test_settings_override_wins_over_backend_default(tmp_path):
    settings = Settings(data_dir=tmp_path, threshold=0.33, margin=0.11)
    assert settings.recognition_defaults() == (0.33, 0.11)
    engine = Engine(settings)
    try:
        assert engine.create().config.threshold == pytest.approx(0.33)
    finally:
        engine.close()


def test_environment_ratio_is_validated(monkeypatch):
    monkeypatch.delenv("ODIN_THRESHOLD", raising=False)
    assert optional_ratio("ODIN_THRESHOLD") is None
    monkeypatch.setenv("ODIN_THRESHOLD", " 0.5 ")
    assert optional_ratio("ODIN_THRESHOLD") == 0.5
    monkeypatch.setenv("ODIN_THRESHOLD", "1.4")
    with pytest.raises(ValueError, match="entre 0 y 1"):
        optional_ratio("ODIN_THRESHOLD")
