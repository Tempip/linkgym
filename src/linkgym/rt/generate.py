"""Trace generation with Sionna RT (needs the ``rt`` extra); used by ``linkgym-traces``.

For every route, the receiver positions of all slots are checked against the scene
geometry (outdoors, not inside a building, away from walls). Paths are then solved at
anchors every ``anchor_spacing_slots`` slots with ``PathSolver(deterministic=True)``, one
receiver per solve; between anchors, the channel evolves with each path's Doppler shift
(``Paths.cfr`` with ``num_time_steps``). Trajectories with a slot of zero gain (no path)
are dropped and counted. Sionna RT is imported only when a function here is called.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from linkgym.rt.config import (
    ConfigError,
    GeneratorConfig,
    Route,
    category_from_los_share,
    split_overlap,
)

NUM_CLEARANCE_DIRECTIONS = 16
SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:purple"}


class RouteError(ValueError):
    """A route leaves open ground: inside a building, too close to a wall or off the scene."""


def import_rt():
    """Import Dr.Jit, Mitsuba and Sionna RT, with a clear message if they are missing."""
    try:
        import drjit as dr
        import mitsuba as mi
        import sionna.rt as rt
    except ImportError as e:
        raise ImportError(
            "linkgym-traces needs Sionna RT. Install the rt extra in its own environment, "
            'e.g. `pip install "linkgym[rt]"` in a new venv (see docs/channels.md). '
            f"Import failed: {e}"
        ) from e
    return dr, mi, rt


# Scene


def scene_file(config: GeneratorConfig) -> str:
    """Path of the scene's Mitsuba XML file: a built-in Sionna RT scene or a file."""
    _, _, rt = import_rt()
    builtin = getattr(rt.scene, config.scene, None)
    if isinstance(builtin, str) and builtin.endswith(".xml"):
        return builtin
    path = os.path.join(config.base_dir, config.scene)
    if not os.path.isfile(path):
        raise ConfigError(
            f"scene {config.scene!r} is neither a built-in Sionna RT scene nor a file ({path})"
        )
    return path


def scene_sha256(xml_path: str) -> str:
    """SHA-256 over the scene XML and every file it references, in sorted order."""
    root = os.path.dirname(os.path.abspath(xml_path))
    tree = ElementTree.parse(xml_path)
    referenced = {
        e.get("value")
        for e in tree.iter("string")
        if e.get("name") == "filename" and e.get("value")
    }
    h = hashlib.sha256()
    for rel in [os.path.basename(xml_path), *sorted(referenced)]:
        with open(os.path.join(root, rel), "rb") as f:
            content = f.read()
        h.update(rel.replace(os.sep, "/").encode() + b"\0")
        h.update(hashlib.sha256(content).digest())
    return h.hexdigest()


@dataclass
class Scene:
    """A loaded Sionna RT scene with the configured transmitter and antennas."""

    config: GeneratorConfig
    xml_path: str
    rt_scene: Any
    tx: Any


def load_scene(config: GeneratorConfig) -> Scene:
    """Load the scene and add the transmitter and the antenna arrays."""
    _, mi, rt = import_rt()
    xml_path = scene_file(config)
    scene = rt.load_scene(xml_path)
    scene.frequency = config.carrier_frequency_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern=config.tx_antenna.pattern,
        polarization=config.tx_antenna.polarization,
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern=config.rx_antenna.pattern,
        polarization=config.rx_antenna.polarization,
    )
    tx = rt.Transmitter(
        name="tx",
        position=mi.Point3f(*config.tx_position),
        orientation=None if config.tx_orientation is None else mi.Point3f(*config.tx_orientation),
        look_at=None if config.tx_look_at is None else mi.Point3f(*config.tx_look_at),
    )
    scene.add(tx)
    return Scene(config=config, xml_path=xml_path, rt_scene=scene, tx=tx)


class Geometry:
    """Ray casts on the scene's Mitsuba geometry, vectorized over points."""

    def __init__(self, scene: Scene) -> None:
        _, self._mi, _ = import_rt()
        self._scene = scene.rt_scene.mi_scene
        bbox = self._scene.bbox()
        self.top = float(bbox.max.z) + 1.0

    def _points(self, x, y, z):
        mi = self._mi
        f = np.float32
        return mi.Point3f(mi.Float(x.astype(f)), mi.Float(y.astype(f)), mi.Float(z.astype(f)))

    def _ray(self, origin, direction, maxt=None):
        mi = self._mi
        d = [mi.Float(np.asarray(c, dtype=np.float32)) for c in direction]
        ray = mi.Ray3f(o=self._points(*origin), d=mi.Vector3f(*d))
        if maxt is not None:
            ray.maxt = mi.Float(np.asarray(maxt, dtype=np.float32))
        return ray

    def ground(self, xy: np.ndarray) -> np.ndarray:
        """Height of the lowest surface below each (x, y) [m]; NaN where there is none."""
        x, y = xy[:, 0], xy[:, 1]
        n = len(xy)
        origin_z = np.full(n, self.top)
        lowest = np.full(n, np.nan)
        active = np.ones(n, dtype=bool)
        for _ in range(64):
            si = self._scene.ray_intersect(
                self._ray((x, y, origin_z), (np.zeros(n), np.zeros(n), -np.ones(n)))
            )
            hit = np.asarray(si.is_valid()) & active
            if not hit.any():
                break
            z = origin_z - np.asarray(si.t)
            lowest = np.where(hit, z, lowest)
            origin_z = np.where(hit, z - 1e-3, origin_z)
            active = hit
        return lowest

    def top_surface(self, xy: np.ndarray) -> np.ndarray:
        """Height of the highest surface at each (x, y) [m]; NaN where there is none."""
        n = len(xy)
        z0 = np.full(n, self.top)
        si = self._scene.ray_intersect(
            self._ray((xy[:, 0], xy[:, 1], z0), (np.zeros(n), np.zeros(n), -np.ones(n)))
        )
        return np.where(np.asarray(si.is_valid()), z0 - np.asarray(si.t), np.nan)

    def covered(self, points: np.ndarray) -> np.ndarray:
        """True where something is above the point: inside a building or under a roof."""
        n = len(points)
        ray = self._ray(points.T, (np.zeros(n), np.zeros(n), np.ones(n)))
        return np.asarray(self._scene.ray_test(ray))

    def clearance(self, points: np.ndarray) -> np.ndarray:
        """Distance [m] to the nearest surface in 16 horizontal directions at each point."""
        n = len(points)
        nearest = np.full(n, np.inf)
        for a in np.arange(NUM_CLEARANCE_DIRECTIONS) * 2 * np.pi / NUM_CLEARANCE_DIRECTIONS:
            direction = (np.full(n, np.cos(a)), np.full(n, np.sin(a)), np.zeros(n))
            si = self._scene.ray_intersect(self._ray(points.T, direction))
            t = np.where(np.asarray(si.is_valid()), np.asarray(si.t), np.inf)
            nearest = np.minimum(nearest, t)
        return nearest

    def line_of_sight(self, points: np.ndarray, target: np.ndarray) -> np.ndarray:
        """True where the segment from the point to ``target`` is unobstructed."""
        delta = target[None, :] - points
        dist = np.linalg.norm(delta, axis=1)
        ray = self._ray(points.T, (delta / dist[:, None]).T, maxt=dist * (1 - 1e-4))
        return ~np.asarray(self._scene.ray_test(ray))


@dataclass
class RouteLayout:
    """Receiver positions of all slots of one route, checked against the geometry."""

    route: Route
    positions: np.ndarray  # [J, N, 3] m
    directions: np.ndarray  # [J, N, 2] unit vectors of travel
    min_clearance: float
    geometric_los: float  # share of slots with an unobstructed line to the transmitter


def layout_route(config: GeneratorConfig, geometry: Geometry, route: Route) -> RouteLayout:
    """Place every slot of ``route`` at receiver height and check it is on open ground."""
    distances = config.slot_distances(route)
    xy, directions = route.points(distances.ravel())
    ground = geometry.ground(xy)
    positions = np.column_stack([xy, ground + config.rx_height_m])

    def bad(mask: np.ndarray, reason: str) -> RouteError:
        where = xy[mask]
        sample = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in where[:: max(1, len(where) // 5)][:5])
        share = mask.mean()
        return RouteError(
            f"route {route.name!r}: {share:.1%} of the slot positions are {reason}, e.g. {sample}"
        )

    if np.isnan(ground).any():
        raise bad(np.isnan(ground), "outside the scene (no ground below)")
    covered = geometry.covered(positions)
    if covered.any():
        raise bad(covered, "inside a building or under a structure")
    clearance = geometry.clearance(positions)
    if (clearance < config.min_clearance_m).any():
        raise bad(
            clearance < config.min_clearance_m,
            f"closer than min_clearance_m = {config.min_clearance_m} m to a surface",
        )
    los = geometry.line_of_sight(positions, np.asarray(config.tx_position))
    shape = distances.shape
    return RouteLayout(
        route=route,
        positions=positions.reshape(*shape, 3),
        directions=directions.reshape(*shape, 2),
        min_clearance=float(clearance.min()),
        geometric_los=float(los.mean()),
    )


# Path solving


def prb_frequencies(config: GeneratorConfig) -> np.ndarray:
    """Baseband frequencies [Hz] at which the channel is computed.

    ``center``: the centre of each PRB; ``mean12``: all 12 subcarriers of each PRB.
    """
    scs = config.subcarrier_spacing_hz
    subcarriers = (np.arange(12 * config.num_prb) - 6 * config.num_prb) * scs
    if config.prb_sampling == "mean12":
        return subcarriers
    return subcarriers.reshape(config.num_prb, 12).mean(axis=1)


class Solver:
    """One receiver, solved at anchors, with the channel evolved by Doppler in between."""

    def __init__(self, scene: Scene) -> None:
        _, self._mi, rt = import_rt()
        from sionna.rt.constants import InteractionType

        self._none = int(InteractionType.NONE)
        self.config = scene.config
        self._scene = scene.rt_scene
        self._rx = rt.Receiver(name="rx", position=self._mi.Point3f(0.0, 0.0, 0.0))
        self._scene.add(self._rx)
        self._solver = rt.PathSolver(deterministic=True)
        self._kwargs = asdict(scene.config.solver)
        self._frequencies = prb_frequencies(scene.config)

    def solve(
        self, position: np.ndarray, velocity: np.ndarray, num_time_steps: int
    ) -> tuple[np.ndarray, int, bool]:
        """Gain [num_time_steps, num_prb] from one anchor, number of paths, and LoS flag."""
        mi = self._mi
        self._rx.position = mi.Point3f(*(float(v) for v in position))
        self._rx.velocity = mi.Vector3f(*(float(v) for v in velocity))
        paths = self._solver(self._scene, **self._kwargs)
        gain = np.abs(self.frequency_response(paths, num_time_steps)) ** 2
        if self.config.prb_sampling == "mean12":
            gain = gain.reshape(num_time_steps, self.config.num_prb, 12).mean(axis=2)
        valid = np.asarray(paths.valid.numpy()).reshape(-1).astype(bool)
        num_paths = int(np.count_nonzero(valid))
        los = False
        if num_paths:
            interactions = np.asarray(paths.interactions.numpy())
            interactions = interactions.reshape(interactions.shape[0], -1)
            los = bool(np.any(valid & np.all(interactions == self._none, axis=0)))
        return gain, num_paths, los

    def frequency_response(self, paths, num_time_steps: int) -> np.ndarray:
        """Channel frequency response [num_time_steps, num_frequencies], complex128.

        The same model as Sionna RT's ``Paths.cir`` and ``Paths.cfr`` (path coefficient
        times carrier phase exp(-j 2 pi f_c tau), Doppler phase exp(j 2 pi f_D t) at
        t = n * slot duration, delay phase exp(-j 2 pi f tau) per frequency), computed in
        float64 with the valid paths sorted by delay, coefficient and Doppler shift.
        ``Paths.cfr`` sums the paths in the order the GPU returns them, which changes from
        process to process; a fixed order makes the result reproducible bit for bit.
        """
        valid = np.asarray(paths.valid.numpy()).reshape(-1).astype(bool)
        a_re, a_im = (np.asarray(x.numpy(), dtype=np.float64).reshape(-1)[valid] for x in paths.a)
        tau = np.asarray(paths.tau.numpy(), dtype=np.float64).reshape(-1)[valid]
        doppler = np.asarray(paths.doppler.numpy(), dtype=np.float64).reshape(-1)[valid]
        order = np.lexsort((doppler, a_im, a_re, tau))
        a, tau, doppler = (a_re + 1j * a_im)[order], tau[order], doppler[order]
        a = a * np.exp(-2j * np.pi * self.config.carrier_frequency_hz * tau)
        delay_phase = np.exp(-2j * np.pi * self._frequencies[:, None] * tau[None, :])
        h = np.empty((num_time_steps, len(self._frequencies)), dtype=np.complex128)
        for n in range(num_time_steps):
            t = n * self.config.slot_duration_s
            h[n] = (delay_phase * (a * np.exp(2j * np.pi * doppler * t))[None, :]).sum(axis=1)
        return h

    def solve_slots(
        self, positions: np.ndarray, directions: np.ndarray, spacing: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gains [N, num_prb] for consecutive slots, solving every ``spacing`` slots."""
        n = len(positions)
        gain = np.empty((n, self.config.num_prb))
        anchors = range(0, n, spacing)
        num_paths = np.empty(len(anchors), dtype=np.int64)
        los = np.empty(len(anchors), dtype=bool)
        for i, start in enumerate(anchors):
            steps = min(spacing, n - start)
            velocity = np.append(directions[start] * self.config.speed_mps, 0.0)
            gain[start : start + steps], num_paths[i], los[i] = self.solve(
                positions[start], velocity, steps
            )
        return gain, num_paths, los


# Commands


def check(config: GeneratorConfig, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Validate all routes against the scene geometry and report their layout."""
    scene = load_scene(config)
    geometry = Geometry(scene)
    overlap = split_overlap(config)
    report = {"routes": [], "splits": {}}
    header = ("route", "split", "length", "traj", "clear", "LoS", "near")
    log("{:24s} {:6s} {:>8s} {:>5s} {:>6s} {:>5s} {:>5s}".format(*header))
    for route in config.routes:
        layout = layout_route(config, geometry, route)
        entry = {
            "name": route.name,
            "split": route.split,
            "group": route.group,
            "length_m": route.length,
            "num_trajectories": config.num_trajectories(route),
            "min_clearance_m": layout.min_clearance,
            "geometric_los_share": layout.geometric_los,
            "near_other_split_share": overlap[route.name],
        }
        report["routes"].append(entry)
        split = report["splits"].setdefault(route.split, {"routes": 0, "length_m": 0.0})
        split["routes"] += 1
        split["length_m"] += route.length
        log(
            f"{route.name:24s} {route.split:6s} {route.length:7.1f}m "
            f"{entry['num_trajectories']:5d} {layout.min_clearance:5.1f}m "
            f"{layout.geometric_los:5.2f} {overlap[route.name]:5.1%}"
        )
    num_anchors = sum(
        e["num_trajectories"] * config.num_slots // config.anchor_spacing_slots
        for e in report["routes"]
    )
    report["num_anchors"] = num_anchors
    for name, split in report["splits"].items():
        log(f"split {name}: {split['routes']} routes, {split['length_m']:.0f} m")
    log(f"{len(config.routes)} routes, {num_anchors} path solves")
    return report


def generate(
    config: GeneratorConfig,
    output: str | os.PathLike,
    *,
    max_trajectories: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Generate the trace file ``output`` (format version 1) from ``config``."""
    from linkgym.channels import write_trace

    dr, mi, _ = import_rt()
    scene = load_scene(config)
    geometry = Geometry(scene)
    layouts = [layout_route(config, geometry, route) for route in config.routes]
    solver = Solver(scene)
    k = config.anchor_spacing_slots

    gains, splits, groups, positions, num_paths, los, summary = [], [], [], [], [], [], []
    categories = []
    start_all = time.perf_counter()
    for layout in layouts:
        route = layout.route
        num_traj = len(layout.positions)
        if max_trajectories is not None:
            num_traj = min(num_traj, max_trajectories)
        start = time.perf_counter()
        kept = 0
        dropped = []  # trajectory index along the route, reason, mean gain
        los_anchors = []
        for j in range(num_traj):
            gain, paths, line = solver.solve_slots(layout.positions[j], layout.directions[j], k)
            mean_gain_db = 10 * np.log10(gain.mean()) if gain.mean() > 0 else -np.inf
            reason = None
            if np.any(gain.sum(axis=1) == 0):
                reason = "zero_gain"  # a slot without any path
            elif config.min_mean_gain_db is not None and mean_gain_db < config.min_mean_gain_db:
                reason = "low_mean_gain"
            if reason:
                dropped.append(
                    {
                        "trajectory": j,
                        "reason": reason,
                        "mean_gain_db": round(float(mean_gain_db), 2)
                        if np.isfinite(mean_gain_db)
                        else None,
                    }
                )
                continue
            kept += 1
            gains.append(gain.astype(np.float32))
            splits.append(route.split)
            groups.append(route.group)
            positions.append(layout.positions[j].astype(np.float32))
            num_paths.append(paths)
            los.append(line)
            los_anchors.append(line)
        dr.sync_thread()
        elapsed = time.perf_counter() - start
        anchors = num_traj * config.num_slots // k
        los_share = float(np.mean(los_anchors)) if los_anchors else None
        category = route.category
        if category is None and los_anchors:
            category = category_from_los_share(los_share)
        categories.extend([category] * kept)
        summary.append(
            {
                "name": route.name,
                "split": route.split,
                "group": route.group,
                "category": category,
                "category_source": "config" if route.category else "los_share",
                "length_m": round(route.length, 3),
                "num_trajectories": kept,
                "num_dropped": len(dropped),
                "num_dropped_zero_gain": sum(d["reason"] == "zero_gain" for d in dropped),
                "num_dropped_low_mean_gain": sum(d["reason"] == "low_mean_gain" for d in dropped),
                "dropped": dropped,
                "los_share": None if los_share is None else round(los_share, 6),
                "waypoints": [list(p) for p in route.waypoints],
            }
        )
        share = "n/a" if los_share is None else f"{los_share:.2f}"
        log(
            f"{route.name}: {kept} trajectories kept, {len(dropped)} dropped "
            f"({summary[-1]['num_dropped_zero_gain']} zero gain, "
            f"{summary[-1]['num_dropped_low_mean_gain']} low mean gain), "
            f"LoS share {share} ({category}), {elapsed:.0f} s "
            f"({1e3 * elapsed / max(anchors, 1):.0f} ms per path solve)"
        )
    total_time = time.perf_counter() - start_all
    if not gains:
        raise RuntimeError("all trajectories were dropped; no trace written")

    num_dropped = sum(s["num_dropped"] for s in summary)
    attrs = {
        "scene": config.scene,
        "scene_sha256": scene_sha256(scene.xml_path),
        "tx_position": np.asarray(config.tx_position),
        "tx_orientation": np.asarray(scene.tx.orientation, dtype=np.float64).reshape(-1)[:3],
        "tx_antenna": json.dumps(asdict(config.tx_antenna)),
        "rx_antenna": json.dumps(asdict(config.rx_antenna)),
        "rx_height_m": config.rx_height_m,
        "speed_mps": config.speed_mps,
        "anchor_spacing_slots": k,
        "anchor_spacing_m": config.anchor_spacing_m,
        "solver": json.dumps(
            asdict(config.solver) | {"deterministic": True, "synthetic_array": True},
            sort_keys=True,
        ),
        "mitsuba_variant": mi.variant(),
        "versions": json.dumps(_versions(), sort_keys=True),
        "routes": json.dumps(summary),
        "num_dropped": num_dropped,
        "min_mean_gain_db": np.nan if config.min_mean_gain_db is None else config.min_mean_gain_db,
        "generator_config": json.dumps(config.to_dict(), sort_keys=True),
    }
    if config.attribution_text:
        attrs["attribution"] = config.attribution_text
    write_trace(
        output,
        np.stack(gains),
        splits,
        carrier_frequency_hz=config.carrier_frequency_hz,
        subcarrier_spacing_hz=config.subcarrier_spacing_hz,
        slot_duration_s=config.slot_duration_s,
        prb_sampling=config.prb_sampling,
        group=np.asarray(groups),
        rx_position=np.stack(positions),
        num_paths=np.stack(num_paths),
        los=np.stack(los),
        category=categories,
        attrs=attrs,
    )
    log(
        f"wrote {output}: {len(gains)} trajectories x {config.num_slots} slots, "
        f"{num_dropped} dropped, {total_time:.0f} s"
    )
    return {"routes": summary, "num_dropped": num_dropped, "seconds": total_time}


def accuracy(
    config: GeneratorConfig,
    route_name: str,
    *,
    start_m: float = 0.0,
    num_slots: int = 1000,
    spacings: tuple[int, ...] = (5, 10, 20, 50),
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Compare anchored channels with a path solve at every slot on a stretch of a route."""
    config = config.select([route_name])
    route = config.routes[0]
    distances = start_m + np.arange(num_slots) * config.slot_length_m
    if distances[-1] > route.length:
        raise ConfigError(f"route {route_name!r} is {route.length:.1f} m, the stretch ends later")
    dr, _, _ = import_rt()
    scene = load_scene(config)
    geometry = Geometry(scene)
    xy, directions = route.points(distances)
    positions = np.column_stack([xy, geometry.ground(xy) + config.rx_height_m])
    solver = Solver(scene)
    solver.solve(positions[0], np.zeros(3), 1)  # warm-up
    dr.sync_thread()

    start = time.perf_counter()
    reference, paths, los = solver.solve_slots(positions, directions, 1)
    dr.sync_thread()
    result = {
        "route": route_name,
        "start_m": start_m,
        "num_slots": num_slots,
        "length_m": num_slots * config.slot_length_m,
        "los_share": float(los.mean()),
        "num_paths": [int(paths.min()), int(np.median(paths)), int(paths.max())],
        "reference_seconds": time.perf_counter() - start,
        "spacings": {},
    }
    log(
        f"{route_name}, {start_m:.1f}-{start_m + result['length_m']:.1f} m: LoS share "
        f"{result['los_share']:.2f}, paths min/median/max {result['num_paths']}, "
        f"reference {result['reference_seconds']:.0f} s"
    )
    ref_db = 10 * np.log10(reference)
    ref_wb = 10 * np.log10(reference.mean(axis=1))
    for spacing in spacings:
        start = time.perf_counter()
        gain, _, _ = solver.solve_slots(positions, directions, spacing)
        dr.sync_thread()
        seconds = time.perf_counter() - start
        err = np.abs(10 * np.log10(gain) - ref_db)
        wb = np.abs(10 * np.log10(gain.mean(axis=1)) - ref_wb)
        entry = {
            "spacing_m": spacing * config.slot_length_m,
            "seconds": seconds,
            "nmse_db": float(10 * np.log10(np.sum((gain - reference) ** 2) / np.sum(reference**2))),
            "prb_gain_error_db_median": float(np.median(err)),
            "prb_gain_error_db_p95": float(np.percentile(err, 95)),
            "wideband_error_db_mean": float(wb.mean()),
            "wideband_error_db_max": float(wb.max()),
            "wideband_error_share_above_1db": float(np.mean(wb > 1.0)),
        }
        result["spacings"][spacing] = entry
        log(
            f"  every {spacing:3d} slots ({entry['spacing_m'] * 100:.1f} cm): "
            f"NMSE {entry['nmse_db']:6.1f} dB, per-PRB gain error median "
            f"{entry['prb_gain_error_db_median']:.2f} / p95 "
            f"{entry['prb_gain_error_db_p95']:.2f} dB, "
            f"wideband error mean {entry['wideband_error_db_mean']:.2f} / max "
            f"{entry['wideband_error_db_max']:.2f} dB, slots off by > 1 dB "
            f"{entry['wideband_error_share_above_1db']:.1%}, {seconds:.1f} s"
        )
    return result


def plot(
    config: GeneratorConfig,
    output: str | os.PathLike,
    *,
    margin_m: float = 40.0,
    resolution_m: float = 1.0,
    labels: bool = False,
) -> None:
    """Top view: buildings shaded by height, transmitter, routes coloured by split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene = load_scene(config)
    geometry = Geometry(scene)
    points = np.concatenate(
        [np.asarray(r.waypoints) for r in config.routes] + [np.asarray([config.tx_position[:2]])]
    )
    x0, y0 = points.min(axis=0) - margin_m
    x1, y1 = points.max(axis=0) + margin_m
    xs = np.arange(x0, x1, resolution_m) + resolution_m / 2
    ys = np.arange(y0, y1, resolution_m) + resolution_m / 2
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    height = (geometry.top_surface(grid) - geometry.ground(grid)).reshape(len(ys), len(xs))
    buildings = np.where(height > 0.5, height, np.nan)

    fig, ax = plt.subplots(figsize=(10, 10 * (y1 - y0) / (x1 - x0)))
    image = ax.imshow(
        buildings,
        origin="lower",
        extent=(x0, x1, y0, y1),
        cmap="Greys",
        vmin=0,
        vmax=np.nanpercentile(buildings, 99) * 1.3,
    )
    fig.colorbar(image, ax=ax, shrink=0.6, label="building height [m]")
    lengths: dict[str, float] = {}
    for route in config.routes:
        xy = np.asarray(route.waypoints)
        color = SPLIT_COLORS.get(route.split, "tab:green")
        ax.plot(xy[:, 0], xy[:, 1], "-", color=color, lw=2.2, solid_capstyle="round")
        lengths[route.split] = lengths.get(route.split, 0.0) + route.length
        if labels:
            mid = route.points(np.asarray([route.length / 2]))[0][0]
            ax.annotate(route.name, mid, fontsize=7, xytext=(3, 3), textcoords="offset points")
    order = {name: i for i, name in enumerate(SPLIT_COLORS)}
    for split, length in sorted(lengths.items(), key=lambda kv: order.get(kv[0], len(order))):
        n = sum(r.split == split for r in config.routes)
        ax.plot(
            [],
            [],
            "-",
            color=SPLIT_COLORS.get(split, "tab:green"),
            lw=2.2,
            label=f"{split}: {n} routes, {length:.0f} m",
        )
    ax.plot(*config.tx_position[:2], "*", color="tab:red", ms=16, mec="k", label="transmitter")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{config.scene}: routes by split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8)
    ax.set_aspect("equal")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("linkgym", "sionna-rt", "mitsuba", "drjit", "numpy", "h5py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions
