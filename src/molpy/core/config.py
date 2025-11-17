import contextlib
import threading
from collections.abc import Generator
from typing import Self

from pydantic import BaseModel, Field


class Config(BaseModel):
    log_level: str = Field(default="INFO")

    n_threads: int = Field(
        default=1, description="Number of threads to use for parallel computations"
    )

    # --- Internal singleton & lock ---
    _instance: Self = None
    _lock = threading.Lock()

    # --- Singleton accessor ---
    @classmethod
    def instance(cls) -> Self:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # --- Update config ---
    @classmethod
    def update(cls, **kwargs):
        with cls._lock:
            cls._instance = cls.instance().model_copy(update=kwargs)

    # --- Reset to default ---
    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = cls()

    # --- Context manager for temporary override ---
    @classmethod
    @contextlib.contextmanager
    def temporary(cls, **overrides) -> Generator[None, None, None]:
        with cls._lock:
            old = cls.instance().model_copy()
            cls._instance = cls._instance.model_copy(update=overrides)
        try:
            yield
        finally:
            with cls._lock:
                cls._instance = old

    def __repr__(self) -> str:
        return f"<MolpyConfig: {self}>"


config = Config.instance()


def get_config() -> Config:
    return config
