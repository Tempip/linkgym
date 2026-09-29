"""Gymnasium environments for 5G NR link adaptation built on NVIDIA Sionna SYS."""

from importlib.metadata import version

import gymnasium

from linkgym.config import ENV_ID, ScenarioConfig
from linkgym.evaluation import evaluate

__version__ = version("linkgym")
__all__ = ["ENV_ID", "ScenarioConfig", "__version__", "evaluate"]

# String entry point: importing linkgym does not import torch or Sionna
if ENV_ID not in gymnasium.registry:
    gymnasium.register(id=ENV_ID, entry_point="linkgym.env:LinkAdaptationEnv")
