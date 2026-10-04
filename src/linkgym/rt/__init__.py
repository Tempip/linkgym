"""Channel trace generation with Sionna RT: the ``linkgym-traces`` command.

Generation needs the ``rt`` extra, installed in its own environment (see docs/channels.md).
Importing this package or :mod:`linkgym.rt.config` does not import Sionna RT;
:mod:`linkgym.rt.generate` imports it when one of its functions is called.

Stability: the ``linkgym-traces`` command is stable; the Python API of ``linkgym.rt``
is experimental (docs/api_stability.md).
"""

from linkgym.rt.config import ConfigError, GeneratorConfig, Route, load_config

__all__ = ["ConfigError", "GeneratorConfig", "Route", "load_config"]
