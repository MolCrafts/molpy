import io

import mollog
import pytest

from molpy.core.config import Config
from molpy.core.logger import _LOG_FORMAT, get_logger


@pytest.fixture(autouse=True)
def _reset_config():
    Config.reset()
    yield
    Config.reset()


class TestGetLogger:
    def test_returns_mollog_logger(self):
        logger = get_logger("molpy.test.basic")
        assert isinstance(logger, mollog.Logger)
        assert logger.name == "molpy.test.basic"

    def test_is_mollog_managed(self):
        # Same name resolves to the same manager-owned logger.
        assert get_logger("molpy.test.mgr") is mollog.get_logger("molpy.test.mgr")

    def test_log_methods_do_not_raise(self):
        logger = get_logger("molpy.test.methods")
        logger.debug("d")
        logger.info("i")
        logger.warning("w")
        logger.error("e")

    def test_emits_through_attached_handler(self):
        logger = get_logger("molpy.test.emit")
        buf = io.StringIO()
        handler = mollog.StreamHandler(stream=buf)
        logger.add_handler(handler)
        try:
            logger.warning("captured message")
        finally:
            logger.remove_handler(handler)
        assert "captured message" in buf.getvalue()

    def test_log_format_constant(self):
        assert _LOG_FORMAT == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def test_root_is_configured(self):
        get_logger("molpy.test.cfg")
        root = mollog.getLogger(None)
        assert root.handlers  # configure() attached an output handler


class TestLoggerLevelFollowsConfig:
    def test_level_from_config(self):
        with Config.temporary(log_level="ERROR"):
            logger = get_logger("molpy.test.level.error")
            assert not logger.is_enabled_for(mollog.Level.INFO)
            assert logger.is_enabled_for(mollog.Level.ERROR)

    def test_default_level_is_info(self):
        logger = get_logger("molpy.test.level.default")
        assert logger.is_enabled_for(mollog.Level.INFO)
        assert not logger.is_enabled_for(mollog.Level.DEBUG)
