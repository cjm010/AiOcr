from src.doc_ai.config import get_settings


def test_settings_build_data_dir(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", "data")

    settings = get_settings()

    assert str(settings.data_dir).endswith("data/test")
    assert settings.app_env == "test"
