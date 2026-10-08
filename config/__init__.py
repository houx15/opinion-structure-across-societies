"""Project configuration: ``config/config.py`` if present, else the example.

    from config import cfg
    cfg.OPENAI_API_KEY
"""

import importlib

try:
    cfg = importlib.import_module("config.config")
except ModuleNotFoundError:
    cfg = importlib.import_module("config.config_example")
