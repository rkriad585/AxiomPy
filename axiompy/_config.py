from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tomllib

_instance: Optional['AxiomConfig'] = None


@dataclass
class AxiomConfig:
    """Global configuration singleton for AxiomPy.

    Attributes:
        precision (int): Number of decimal places for display rounding.
        dtype (str): Default NumPy dtype string (e.g. ``"float64"``).
        verbose (bool): Enable verbose/debug logging.
    """

    precision: int = 6
    dtype: str = "float64"
    verbose: bool = False

    @classmethod
    def load(cls) -> 'AxiomConfig':
        """Load the config singleton, optionally merging values from ``pyproject.toml``.

        Returns:
            AxiomConfig: The singleton instance.
        """
        global _instance
        if _instance is not None:
            return _instance
        cfg = cls()
        try:
            path = Path("pyproject.toml")
            if path.exists():
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                tool = data.get("tool", {}).get("axiompy", {})
                if "precision" in tool:
                    cfg.precision = int(tool["precision"])
                if "dtype" in tool:
                    cfg.dtype = tool["dtype"]
                if "verbose" in tool:
                    cfg.verbose = bool(tool["verbose"])
        except Exception:
            pass
        _instance = cfg
        return cfg

    @classmethod
    def reset(cls):
        """Reset the singleton instance to ``None``.

        The next call to :meth:`load` will re-read ``pyproject.toml``.
        """
        global _instance
        _instance = None

    @classmethod
    def configure(cls, **kwargs):
        """Override config attributes after load.

        Args:
            **kwargs: Key-value pairs matching :class:`AxiomConfig` field names.
        """
        cfg = cls.load()
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
