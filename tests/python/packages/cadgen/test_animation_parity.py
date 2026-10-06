"""Every renderer draws, between keys, what cadgen's baker assumed it would.

The baker drops a key only when ITS interpolation (``_pose_at`` for a transform,
a lerp for an opacity) rebuilds the dropped samples within tolerance, so the
guarantee holds only while @text-to-cad/core's ``animationRuntime.js`` interpolates
identically. ``animationRuntime.parity.json`` beside it holds baked keyframes and
the frames a renderer must draw from them at times between keys: this test pins
the Python half, ``animationRuntime.parity.test.js`` the JavaScript half.

An intended interpolation change regenerates the expected frames (the keys stay):

    .venv/bin/python tests/python/packages/cadgen/test_animation_parity.py
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from cadgen._internal import animation_bake as ab

FIXTURE = Path(__file__).resolve().parents[4] / "packages" / "core" / "src" / "common" / "animationRuntime.parity.json"


def _at(times: list[float], t: float) -> tuple[int, float]:
    last = len(times) - 1
    if t >= times[last]:
        return last, 0.0
    j = next(index for index, value in enumerate(times) if value > t)
    return j - 1, (t - times[j - 1]) / (times[j] - times[j - 1])


def frame(clip: dict, t: float) -> dict:
    """What a renderer draws for ``clip`` at ``t``, with the baker's interpolation."""
    local = t % clip["duration"] if clip["loop"] else min(max(t, 0.0), clip["duration"])
    out: dict = {"matrices": {}, "styles": {}}
    for track in clip["tracks"]:
        i, u = _at(track["times"], local)
        if "transform" in track:
            a = track["transform"][i]
            if u > 0:
                d, r = ab._pose_at(a, track["transform"][i + 1], track["times"][i + 1] - track["times"][i], u)
            else:
                n = math.sqrt(sum(c * c for c in a[3:7]))
                d, r = a[:3], ab._rotation_of([c / n for c in a[3:7]])
            pivot = track["pivot"]
            turned = ab._apply_point(r, (0.0, 0.0, 0.0), pivot)
            move = [pivot[n] + d[n] - turned[n] for n in range(3)]
            rows = [[r[0], r[1], r[2], move[0]], [r[3], r[4], r[5], move[1]], [r[6], r[7], r[8], move[2]], [0.0, 0.0, 0.0, 1.0]]
            for target in track["targets"]:
                out["matrices"][target] = rows
        elif "opacity" in track:
            a = track["opacity"][i]
            b = track["opacity"][i + 1] if u > 0 else None
            if a is not None:
                for target in track["targets"]:
                    out["styles"].setdefault(target, {})["opacity"] = a if b is None else a + (b - a) * u
        elif "visible" in track:
            if track["visible"][i] is not None:
                for target in track["targets"]:
                    out["styles"].setdefault(target, {})["visible"] = track["visible"][i]
    return out


def _close(test: unittest.TestCase, got: object, want: object, where: str) -> None:
    if isinstance(want, dict):
        test.assertEqual(sorted(got), sorted(want), where)  # type: ignore[arg-type]
        for key in want:
            _close(test, got[key], want[key], f"{where}.{key}")  # type: ignore[index]
    elif isinstance(want, list):
        test.assertEqual(len(got), len(want), where)  # type: ignore[arg-type]
        for index, (g, w) in enumerate(zip(got, want)):  # type: ignore[arg-type]
            _close(test, g, w, f"{where}[{index}]")
    elif isinstance(want, bool) or isinstance(want, str):
        test.assertEqual(got, want, where)
    else:
        test.assertAlmostEqual(got, want, delta=1e-9, msg=where)


class TheBakersInterpolationIsTheRenderers(unittest.TestCase):
    def test_every_probe_frame_is_what_the_baker_interpolates(self) -> None:
        parity = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertIsNotNone(ab.normalize_baked_animation(parity["animation"]))
        clips = {clip["id"]: clip for clip in parity["animation"]["clips"]}
        # Every channel and both interpolants are exercised, or the file proves less than it says.
        channels = {name for clip in clips.values() for track in clip["tracks"] for name in ab.CHANNELS if name in track}
        self.assertEqual(channels, set(ab.CHANNELS))
        for probe in parity["probes"]:
            with self.subTest(clip=probe["clip"], t=probe["t"]):
                got = frame(clips[probe["clip"]], probe["t"])
                for part in ("matrices", "styles"):
                    _close(self, got[part], probe[part], f"{probe['clip']}@{probe['t']:.4f}.{part}")


if __name__ == "__main__":
    parity = json.loads(FIXTURE.read_text(encoding="utf-8"))
    clips = {clip["id"]: clip for clip in parity["animation"]["clips"]}
    parity["probes"] = [{"clip": p["clip"], "t": p["t"], **frame(clips[p["clip"]], p["t"])} for p in parity["probes"]]
    FIXTURE.write_text(json.dumps(parity, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"rewrote the expected frames in {FIXTURE}")
