import pytest

from molpy.core.config import Config, config, get_config


@pytest.fixture(autouse=True)
def _reset_config():
    """Each test starts and ends from the documented defaults."""
    Config.reset()
    yield
    Config.reset()


class TestConfigDefaults:
    def test_default_values(self):
        assert config.log_level == "INFO"
        assert config.n_threads == 1

    def test_get_config_returns_singleton(self):
        assert get_config() is config
        assert Config.instance() is config

    def test_is_molcfg_config(self):
        from molcfg import Config as MolcfgConfig

        assert isinstance(config, MolcfgConfig)

    def test_to_dict(self):
        assert config.to_dict() == {"log_level": "INFO", "n_threads": 1}

    def test_dotted_path_access(self):
        assert config["log_level"] == "INFO"


class TestConfigUpdate:
    def test_update_changes_values(self):
        Config.update(log_level="DEBUG", n_threads=8)
        assert config.log_level == "DEBUG"
        assert config.n_threads == 8

    def test_update_visible_through_module_reference(self):
        # The module-level `config` reference reflects updates in place — no
        # stale copy after Config.update().
        Config.update(n_threads=16)
        assert get_config().n_threads == 16
        assert config.n_threads == 16


class TestConfigReset:
    def test_reset_restores_defaults(self):
        Config.update(log_level="ERROR", n_threads=4)
        Config.reset()
        assert config.log_level == "INFO"
        assert config.n_threads == 1

    def test_reset_removes_runtime_keys(self):
        Config.update(extra_key="value")
        assert config.get("extra_key") == "value"
        Config.reset()
        assert config.get("extra_key") is None
        assert set(config.keys()) == {"log_level", "n_threads"}


class TestConfigTemporary:
    def test_temporary_override_and_restore(self):
        Config.update(log_level="DEBUG")
        with Config.temporary(log_level="WARNING"):
            assert config.log_level == "WARNING"
        assert config.log_level == "DEBUG"

    def test_temporary_restores_on_exception(self):
        Config.update(log_level="INFO")
        with pytest.raises(RuntimeError):
            with Config.temporary(log_level="ERROR"):
                assert config.log_level == "ERROR"
                raise RuntimeError("boom")
        assert config.log_level == "INFO"
