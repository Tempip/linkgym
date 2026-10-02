"""Trace generator configuration (a JSON file) and route geometry. Does not import Sionna RT.

A configuration describes one trace file: the scene, the transmitter, the PRB grid, the
receiver motion, the path solver settings and the routes. Each route is a polyline of
(x, y) waypoints [m] at a fixed height above ground, with a split label and a group id.
The receiver moves along it at ``speed_mps``; the route is cut into trajectories of
``num_slots`` consecutive slots (the remainder is dropped). See docs/channels.md.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

OSM_SCENES = ("munich", "etoile", "florence", "san_francisco")
OSM_ATTRIBUTION = (
    "Scene built from OpenStreetMap data, (c) OpenStreetMap contributors, available under "
    "the Open Data Commons Open Database License (ODbL), https://www.openstreetmap.org/copyright"
)
ANTENNA_PATTERNS = ("iso", "dipole", "hw_dipole", "tr38901")
POLARIZATIONS = ("V", "H")  # one port: SISO
PRB_SAMPLINGS = ("center", "mean12")
SUBCARRIER_SPACINGS = (15e3, 30e3, 60e3, 120e3, 240e3)
CATEGORIES = ("los", "nlos", "transition")
# A route without a configured category gets one from its LoS share (share of path solves
# with a line-of-sight path): "nlos" up to NLOS_MAX_LOS_SHARE, "los" from LOS_MIN_LOS_SHARE
NLOS_MAX_LOS_SHARE = 0.05
LOS_MIN_LOS_SHARE = 0.75


class ConfigError(ValueError):
    """The generator configuration is invalid."""


@dataclass(frozen=True)
class Antenna:
    """Single-element antenna (SISO): a Sionna RT pattern and one polarization."""

    pattern: str = "iso"
    polarization: str = "V"


@dataclass(frozen=True)
class SolverSettings:
    """Arguments of Sionna RT's ``PathSolver.__call__``; the solver is always deterministic."""

    max_depth: int = 10
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True
    diffraction: bool = True
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True
    samples_per_src: int = 1_000_000
    max_num_paths_per_src: int = 1_000_000
    seed: int = 42


@dataclass(frozen=True)
class Route:
    """A receiver route: a polyline of (x, y) waypoints [m] with a split label and a group.

    ``category`` ("los", "nlos" or "transition") is optional; without it, the generator
    derives one from the route's LoS share, see :func:`category_from_los_share`.
    """

    name: str
    split: str
    group: int
    waypoints: tuple[tuple[float, float], ...]
    category: str | None = None

    @property
    def length(self) -> float:
        """Length of the polyline [m]."""
        return float(np.sum(_segment_lengths(np.asarray(self.waypoints))))

    def points(self, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions [n, 2] and unit directions of travel [n, 2] at arc lengths [m].

        At a waypoint, the direction is that of the segment that starts there.
        """
        wp = np.asarray(self.waypoints, dtype=np.float64)
        seg = np.diff(wp, axis=0)
        seg_len = _segment_lengths(wp)
        start = np.concatenate([[0.0], np.cumsum(seg_len)])
        distances = np.asarray(distances, dtype=np.float64)
        k = np.clip(np.searchsorted(start, distances, side="right") - 1, 0, len(seg) - 1)
        t = (distances - start[k]) / seg_len[k]
        return wp[k] + seg[k] * t[:, None], seg[k] / seg_len[k, None]


@dataclass(frozen=True)
class GeneratorConfig:
    """A trace generator configuration; read it with :func:`load_config`."""

    scene: str
    tx_position: tuple[float, float, float]
    routes: tuple[Route, ...]
    tx_orientation: tuple[float, float, float] | None = None
    tx_look_at: tuple[float, float, float] | None = None
    tx_antenna: Antenna = field(default_factory=Antenna)
    rx_antenna: Antenna = field(default_factory=Antenna)
    rx_height_m: float = 1.5
    carrier_frequency_hz: float = 3.5e9
    subcarrier_spacing_hz: float = 30e3
    num_prb: int = 52
    prb_sampling: str = "center"
    speed_mps: float = 15.0
    num_slots: int = 4000
    anchor_spacing_slots: int = 10
    min_clearance_m: float = 1.0
    min_mean_gain_db: float | None = None
    solver: SolverSettings = field(default_factory=SolverSettings)
    attribution: str | None = None
    description: str | None = None
    base_dir: str = "."  # directory of the configuration file: relative scene paths

    @property
    def slot_duration_s(self) -> float:
        """Slot duration of the 5G NR numerology [s]."""
        return 1e-3 * 15e3 / self.subcarrier_spacing_hz

    @property
    def slot_length_m(self) -> float:
        """Distance travelled per slot [m]."""
        return self.speed_mps * self.slot_duration_s

    @property
    def anchor_spacing_m(self) -> float:
        """Distance between path solves [m]."""
        return self.anchor_spacing_slots * self.slot_length_m

    @property
    def attribution_text(self) -> str | None:
        """The attribution to store: the configured one, or the OSM one for city scenes."""
        if self.attribution is not None:
            return self.attribution
        return OSM_ATTRIBUTION if self.scene in OSM_SCENES else None

    def num_trajectories(self, route: Route) -> int:
        """Number of whole trajectories of ``num_slots`` slots that fit on ``route``."""
        slots = math.floor(route.length / self.slot_length_m * (1 + 1e-12)) + 1
        return slots // self.num_slots

    def slot_distances(self, route: Route) -> np.ndarray:
        """Arc length [m] of every slot of every trajectory of ``route``, shape [J, N]."""
        n = np.arange(self.num_trajectories(route) * self.num_slots, dtype=np.float64)
        return (n * self.slot_length_m).reshape(-1, self.num_slots)

    def to_dict(self) -> dict[str, Any]:
        """The configuration as a JSON-serializable dict, in the file's layout."""
        transmitter: dict[str, Any] = {"position": list(self.tx_position)}
        if self.tx_orientation is not None:
            transmitter["orientation"] = list(self.tx_orientation)
        if self.tx_look_at is not None:
            transmitter["look_at"] = list(self.tx_look_at)
        transmitter["antenna"] = asdict(self.tx_antenna)
        out: dict[str, Any] = {
            "scene": self.scene,
            "attribution": self.attribution,
            "description": self.description,
            "carrier_frequency_hz": self.carrier_frequency_hz,
            "subcarrier_spacing_hz": self.subcarrier_spacing_hz,
            "num_prb": self.num_prb,
            "prb_sampling": self.prb_sampling,
            "transmitter": transmitter,
            "receiver": {"height_m": self.rx_height_m, "antenna": asdict(self.rx_antenna)},
            "speed_mps": self.speed_mps,
            "num_slots": self.num_slots,
            "anchor_spacing_slots": self.anchor_spacing_slots,
            "min_clearance_m": self.min_clearance_m,
            "min_mean_gain_db": self.min_mean_gain_db,
            "solver": asdict(self.solver),
            "routes": [
                {
                    "name": r.name,
                    "split": r.split,
                    "group": r.group,
                    **({"category": r.category} if r.category else {}),
                    "waypoints": [list(p) for p in r.waypoints],
                }
                for r in self.routes
            ],
        }
        return {k: v for k, v in out.items() if v is not None}

    def select(self, names: list[str] | None) -> GeneratorConfig:
        """The same configuration restricted to the routes named in ``names`` (all if None)."""
        if not names:
            return self
        known = {r.name for r in self.routes}
        unknown = [n for n in names if n not in known]
        if unknown:
            raise ConfigError(f"unknown route(s) {unknown}; routes: {sorted(known)}")
        routes = tuple(r for r in self.routes if r.name in names)
        return GeneratorConfig(**{**self._fields(), "routes": routes})

    def select_splits(self, splits: list[str] | None) -> GeneratorConfig:
        """The same configuration restricted to the routes of ``splits`` (all if None)."""
        if not splits:
            return self
        known = {r.split for r in self.routes}
        unknown = [s for s in splits if s not in known]
        if unknown:
            raise ConfigError(f"unknown split(s) {unknown}; splits: {sorted(known)}")
        return self.select([r.name for r in self.routes if r.split in splits])

    def with_solver(self, overrides: dict[str, Any]) -> GeneratorConfig:
        """The same configuration with some solver settings replaced (validated)."""
        if not overrides:
            return self
        merged = asdict(self.solver) | overrides
        return GeneratorConfig(**{**self._fields(), "solver": _solver(merged)})

    def _fields(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_config(path: str | os.PathLike) -> GeneratorConfig:
    """Read and validate a generator configuration file (JSON)."""
    path = os.fspath(path)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"{path}: cannot read the configuration ({e})") from e
    try:
        return parse_config(data, base_dir=os.path.dirname(os.path.abspath(path)))
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None


def parse_config(data: dict[str, Any], base_dir: str = ".") -> GeneratorConfig:
    """Validate a configuration given as a dict (the parsed JSON file)."""
    top = _Section(data, "configuration")
    top.allow(
        "scene",
        "attribution",
        "description",
        "carrier_frequency_hz",
        "subcarrier_spacing_hz",
        "num_prb",
        "prb_sampling",
        "transmitter",
        "receiver",
        "speed_mps",
        "num_slots",
        "anchor_spacing_slots",
        "min_clearance_m",
        "min_mean_gain_db",
        "solver",
        "routes",
    )
    scene = top.string("scene", required=True)

    tx = _Section(top.get("transmitter", required=True), "transmitter")
    tx.allow("position", "orientation", "look_at", "antenna")
    tx_position = tx.vector("position", required=True)
    tx_orientation = tx.vector("orientation")
    tx_look_at = tx.vector("look_at")
    if tx_orientation is not None and tx_look_at is not None:
        raise ConfigError("transmitter: give either 'orientation' or 'look_at', not both")

    rx = _Section(top.get("receiver", {}), "receiver")
    rx.allow("height_m", "antenna")

    defaults = GeneratorConfig(scene="", tx_position=(0.0, 0.0, 0.0), routes=())
    config = GeneratorConfig(
        scene=scene,
        tx_position=tx_position,
        tx_orientation=tx_orientation,
        tx_look_at=tx_look_at,
        tx_antenna=_antenna(tx.get("antenna", {}), "transmitter.antenna"),
        rx_antenna=_antenna(rx.get("antenna", {}), "receiver.antenna"),
        rx_height_m=rx.number("height_m", defaults.rx_height_m, positive=True),
        carrier_frequency_hz=top.number(
            "carrier_frequency_hz", defaults.carrier_frequency_hz, positive=True
        ),
        subcarrier_spacing_hz=top.number(
            "subcarrier_spacing_hz", defaults.subcarrier_spacing_hz, positive=True
        ),
        num_prb=top.integer("num_prb", defaults.num_prb, minimum=1),
        prb_sampling=top.choice("prb_sampling", defaults.prb_sampling, PRB_SAMPLINGS),
        speed_mps=top.number("speed_mps", defaults.speed_mps, positive=True),
        num_slots=top.integer("num_slots", defaults.num_slots, minimum=1),
        anchor_spacing_slots=top.integer(
            "anchor_spacing_slots", defaults.anchor_spacing_slots, minimum=1
        ),
        min_clearance_m=top.number("min_clearance_m", defaults.min_clearance_m, minimum=0.0),
        min_mean_gain_db=(
            None if top.get("min_mean_gain_db") is None else top.number("min_mean_gain_db", 0.0)
        ),
        solver=_solver(top.get("solver", {})),
        attribution=top.string("attribution"),
        description=top.string("description"),
        routes=(),
        base_dir=base_dir,
    )
    if config.subcarrier_spacing_hz not in SUBCARRIER_SPACINGS:
        raise ConfigError(
            f"subcarrier_spacing_hz must be one of {SUBCARRIER_SPACINGS} (5G NR numerologies)"
        )
    if config.num_slots % config.anchor_spacing_slots:
        raise ConfigError(
            f"num_slots ({config.num_slots}) must be a multiple of anchor_spacing_slots "
            f"({config.anchor_spacing_slots})"
        )

    raw_routes = top.get("routes", required=True)
    if not isinstance(raw_routes, list) or not raw_routes:
        raise ConfigError("routes: must be a non-empty list")
    routes = tuple(_route(r, i, config) for i, r in enumerate(raw_routes))
    names = [r.name for r in routes]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"routes: duplicate names {duplicates}")
    return GeneratorConfig(**{**config._fields(), "routes": routes})


def category_from_los_share(los_share: float) -> str:
    """Route category from the share of its path solves that have a line-of-sight path."""
    if los_share <= NLOS_MAX_LOS_SHARE:
        return "nlos"
    if los_share >= LOS_MIN_LOS_SHARE:
        return "los"
    return "transition"


def split_overlap(
    config: GeneratorConfig, radius_m: float = 10.0, step_m: float = 1.0
) -> dict[str, float]:
    """Share of each route's length within ``radius_m`` of a route of another split."""
    samples = {}
    for route in config.routes:
        distances = np.arange(0.0, route.length + step_m / 2, step_m)
        samples[route.name] = route.points(np.minimum(distances, route.length))[0]
    shares = {}
    for route in config.routes:
        others = [samples[o.name] for o in config.routes if o.split != route.split]
        if not others:
            shares[route.name] = 0.0
            continue
        other = np.concatenate(others)
        near = np.zeros(len(samples[route.name]), dtype=bool)
        for i in range(0, len(other), 4096):
            d = np.linalg.norm(samples[route.name][:, None] - other[None, i : i + 4096], axis=-1)
            near |= d.min(axis=1) < radius_m
        shares[route.name] = float(near.mean())
    return shares


def _segment_lengths(wp: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(wp, axis=0), axis=1)


def _antenna(data: Any, where: str) -> Antenna:
    section = _Section(data, where)
    section.allow("pattern", "polarization")
    return Antenna(
        pattern=section.choice("pattern", "iso", ANTENNA_PATTERNS),
        polarization=section.choice("polarization", "V", POLARIZATIONS),
    )


def _solver(data: Any) -> SolverSettings:
    section = _Section(data, "solver")
    defaults = SolverSettings()
    section.allow(*(f.name for f in fields(SolverSettings)))
    values = {}
    for f in fields(SolverSettings):
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            values[f.name] = section.boolean(f.name, default)
        else:
            values[f.name] = section.integer(f.name, default, minimum=0)
    if values["samples_per_src"] < 1 or values["max_num_paths_per_src"] < 1:
        raise ConfigError("solver: samples_per_src and max_num_paths_per_src must be >= 1")
    return SolverSettings(**values)


def _route(data: Any, index: int, config: GeneratorConfig) -> Route:
    where = f"routes[{index}]"
    section = _Section(data, where)
    section.allow("name", "split", "group", "category", "waypoints", "description")
    name = section.string("name", required=True)
    where = f"route {name!r}"
    split = section.string("split", required=True)
    group = section.integer("group", index, minimum=0)
    category = section.get("category")
    if category is not None and category not in CATEGORIES:
        raise ConfigError(f"{where}: 'category' must be one of {CATEGORIES}, got {category!r}")
    raw = section.get("waypoints", required=True)
    try:
        wp = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        wp = np.empty(0)
    if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2 or not np.all(np.isfinite(wp)):
        raise ConfigError(f"{where}: waypoints must be a list of at least two [x, y] pairs")
    if np.any(_segment_lengths(wp) <= 0):
        raise ConfigError(f"{where}: consecutive waypoints must differ")
    route = Route(
        name=name,
        split=split,
        group=group,
        waypoints=tuple(map(tuple, wp.tolist())),
        category=category,
    )
    if config.num_trajectories(route) < 1:
        need = config.num_slots * config.slot_length_m
        raise ConfigError(
            f"{where}: {route.length:.1f} m is shorter than one trajectory "
            f"({config.num_slots} slots at {config.speed_mps} m/s = {need:.1f} m)"
        )
    return route


class _Section:
    """Typed access to one object of the configuration, with error messages naming it."""

    def __init__(self, data: Any, where: str) -> None:
        if not isinstance(data, dict):
            raise ConfigError(f"{where}: must be an object")
        self.data = data
        self.where = where

    def allow(self, *keys: str) -> None:
        unknown = sorted(set(self.data) - set(keys))
        if unknown:
            raise ConfigError(f"{self.where}: unknown key(s) {unknown}; allowed: {list(keys)}")

    def get(self, key: str, default: Any = None, *, required: bool = False) -> Any:
        if key not in self.data:
            if required:
                raise ConfigError(f"{self.where}: missing {key!r}")
            return default
        return self.data[key]

    def string(self, key: str, *, required: bool = False) -> str | None:
        value = self.get(key, required=required)
        if value is not None and (not isinstance(value, str) or not value):
            raise ConfigError(f"{self.where}: {key!r} must be a non-empty string")
        return value

    def choice(self, key: str, default: str, options: tuple[str, ...]) -> str:
        value = self.get(key, default)
        if value not in options:
            raise ConfigError(f"{self.where}: {key!r} must be one of {options}, got {value!r}")
        return value

    def boolean(self, key: str, default: bool) -> bool:
        value = self.get(key, default)
        if not isinstance(value, bool):
            raise ConfigError(f"{self.where}: {key!r} must be true or false")
        return value

    def number(
        self, key: str, default: float, *, positive: bool = False, minimum: float | None = None
    ) -> float:
        value = self.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ConfigError(f"{self.where}: {key!r} must be a number")
        if positive and value <= 0:
            raise ConfigError(f"{self.where}: {key!r} must be > 0")
        if minimum is not None and value < minimum:
            raise ConfigError(f"{self.where}: {key!r} must be >= {minimum}")
        return float(value)

    def integer(self, key: str, default: int, *, minimum: int) -> int:
        value = self.get(key, default)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ConfigError(f"{self.where}: {key!r} must be an integer >= {minimum}")
        return value

    def vector(self, key: str, *, required: bool = False) -> tuple[float, float, float] | None:
        value = self.get(key, required=required)
        if value is None:
            return None
        ok = isinstance(value, list) and len(value) == 3
        ok = ok and all(
            not isinstance(v, bool) and isinstance(v, int | float) and math.isfinite(v)
            for v in value
        )
        if not ok:
            raise ConfigError(f"{self.where}: {key!r} must be a list of three numbers")
        return tuple(float(v) for v in value)
