"""Bake a model's clips (``cadgen.animation``) into sidecar keyframes.

A clip runs ONCE, here, when its model builds: ``update(t, m)`` at every sample
time, against the built tree's occurrence table. What ships is data — the
``animation`` section of the model's sidecar — and every renderer
(``@text-to-cad/core`` ``animationRuntime.js``) interpolates it::

    {"clips": [{"id", "label", "duration", "loop", "tracks": [<track>, ...]}, ...]}

The clips keep the order ``animation=`` declares them in; the first is the one
a viewer opens on.

A track drives one CHANNEL of a set of leaf occurrences (document ids) that it
moves identically. ``times`` start at 0, rise strictly, and end at or before
``duration``; each channel carries one value per time:

    transform  [dx, dy, dz, qx, qy, qz, qw,   the track's "pivot" moves by d while
                d'x, d'y, d'z, wx, wy, wz]    the part turns by q about it: the
                                              matrix is T(pivot + d) R(q) T(-pivot).
                                              d' is d's rate (mm/s) and w the
                                              angular velocity (rad/s, world).
                                              Between keys d is a cubic Hermite
                                              curve, and so is the turn from the
                                              first key as a rotation vector -- a
                                              constant spin is exact. The pivot is
                                              the point the motion moves least. The
                                              rest pose is the identity, and a
                                              rigid move premultiplies whatever the
                                              kinematics put there
    opacity    0..1 | null                     lerp between numbers; null is the
                                              material's own, held
    visible    true | false | null             held; null is the rest state

A key interpolation rebuilds within tolerance is dropped (Ramer-Douglas-Peucker
over each track), and a track whose keys are all one value keeps one key. A
transform's tolerance is measured at the corners of the model's bounding box:
two rigid transforms differ by an affine map, whose largest displacement over
the box is at a corner, so the corners bound the error over every point of the
model. A key's rates are the clip's own (central differences of its samples),
so a smooth motion needs keys only where its curve changes character, and a
part turns at most 120 degrees between two kept keys.
"""

from __future__ import annotations

import bisect
import math
from typing import Any, Iterable, Mapping, Sequence

CHANNELS = ("transform", "opacity", "visible")

# Transform error, as a fraction of the model's bounding-box diagonal, under which
# a key is dropped: a tenth of a pixel with the whole model in view, and a pixel
# zoomed in tenfold -- measured at the box's corners, the worst lever arm any
# part has. The floor is the rounding the keys are written at.
TRANSFORM_TOLERANCE = 1e-4
LENGTH_FLOOR = 1e-4
OPACITY_TOLERANCE = 1.0 / 512.0
# A quaternion's sign is chosen to continue the one before it: a part that turns
# further than this between two samples could be turning either way, and its
# keys would not say which.
MAX_SAMPLE_TURN_DEG = 90.0
# Two kept keys turn less than this apart, well short of the half turn at which
# "the turn from one key to the next" stops being one rotation.
MAX_KEY_TURN_DEG = 120.0

_LENGTH_DIGITS = 4
_QUATERNION_DIGITS = 7
_TIME_DIGITS = 6
_OPACITY_DIGITS = 4


class AnimationError(ValueError):
    """A clip that cannot be baked: a bad target, effect or value."""


# --- Rigid transforms ----------------------------------------------------------
#
# A transform is a 12-tuple: the rotation row-major, then the translation. Every
# effect a clip can apply is rigid, which is what lets a key be written as a
# translation and a quaternion and interpolated without shearing.

_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def _is_identity(t: tuple) -> bool:
    return max(abs(a - b) for a, b in zip(t, _IDENTITY)) < 1e-12


def _compose(a: tuple, b: tuple) -> tuple:
    """``a`` after ``b``."""
    a0, a1, a2, a3, a4, a5, a6, a7, a8, ax, ay, az = a
    b0, b1, b2, b3, b4, b5, b6, b7, b8, bx, by, bz = b
    return (
        a0 * b0 + a1 * b3 + a2 * b6, a0 * b1 + a1 * b4 + a2 * b7, a0 * b2 + a1 * b5 + a2 * b8,
        a3 * b0 + a4 * b3 + a5 * b6, a3 * b1 + a4 * b4 + a5 * b7, a3 * b2 + a4 * b5 + a5 * b8,
        a6 * b0 + a7 * b3 + a8 * b6, a6 * b1 + a7 * b4 + a8 * b7, a6 * b2 + a7 * b5 + a8 * b8,
        a0 * bx + a1 * by + a2 * bz + ax, a3 * bx + a4 * by + a5 * bz + ay, a6 * bx + a7 * by + a8 * bz + az,
    )


def _rotation(axis: tuple, degrees: float, origin: tuple) -> tuple:
    x, y, z = axis
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    r = (
        c + x * x * k, x * y * k - z * s, x * z * k + y * s,
        y * x * k + z * s, c + y * y * k, y * z * k - x * s,
        z * x * k - y * s, z * y * k + x * s, c + z * z * k,
    )
    ox, oy, oz = origin
    return r + (
        ox - (r[0] * ox + r[1] * oy + r[2] * oz),
        oy - (r[3] * ox + r[4] * oy + r[5] * oz),
        oz - (r[6] * ox + r[7] * oy + r[8] * oz),
    )


def _quaternion(t: tuple) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a transform's rotation."""
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = t[:9]
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = ((r21 - r12) * s, (r02 - r20) * s, (r10 - r01) * s, 0.25 / s)
    elif r00 > r11 and r00 > r22:
        s = 2.0 * math.sqrt(1.0 + r00 - r11 - r22)
        q = (0.25 * s, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s)
    elif r11 > r22:
        s = 2.0 * math.sqrt(1.0 + r11 - r00 - r22)
        q = ((r01 + r10) / s, 0.25 * s, (r12 + r21) / s, (r02 - r20) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r22 - r00 - r11)
        q = ((r02 + r20) / s, (r12 + r21) / s, 0.25 * s, (r10 - r01) / s)
    n = math.sqrt(sum(c * c for c in q))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def _rotation_of(q: Sequence[float]) -> tuple:
    x, y, z, w = q
    return (
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    )


def _qmul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _turn_vector(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """The rotation vector (axis * radians, world frame) that turns ``a`` into ``b``."""
    x, y, z, w = _qmul(b, (-a[0], -a[1], -a[2], a[3]))
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return (2.0 * x, 2.0 * y, 2.0 * z)
    angle = 2.0 * math.atan2(s, w)
    return (x / s * angle, y / s * angle, z / s * angle)


def _turned(v: Sequence[float], q: Sequence[float]) -> tuple[float, float, float, float]:
    """``q`` turned further by the rotation vector ``v``."""
    angle = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    s = 0.5 if angle < 1e-12 else math.sin(angle / 2.0) / angle
    turned = _qmul((v[0] * s, v[1] * s, v[2] * s, math.cos(angle / 2.0)), q)
    n = math.sqrt(sum(c * c for c in turned))
    return tuple(c / n for c in turned)  # type: ignore[return-value]


def _hermite_basis(u: float) -> tuple[float, float, float, float]:
    u2 = u * u
    u3 = u2 * u
    return 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, 3 * u2 - 2 * u3, u3 - u2


def _pose_at(a: Sequence[float], b: Sequence[float], span: float, u: float) -> tuple[list[float], tuple]:
    """(d, rotation) at fraction ``u`` between two transform keys ``span`` seconds
    apart: what every renderer does. d is a cubic Hermite curve through the keys'
    values and rates; the turn from ``a`` is one through the zero vector and the
    turn to ``b``, with the keys' angular velocities -- so a constant spin is a
    straight line there, and exact."""
    h00, h10, h01, h11 = _hermite_basis(u)
    d = [h00 * a[n] + h10 * span * a[7 + n] + h01 * b[n] + h11 * span * b[7 + n] for n in range(3)]
    turn = _turn_vector(a[3:7], b[3:7])
    v = [h10 * span * a[10 + n] + h01 * turn[n] + h11 * span * b[10 + n] for n in range(3)]
    return d, _rotation_of(_turned(v, a[3:7]))


def _turn_deg(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


# --- Values a clip passes ------------------------------------------------------


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        try:
            value = float(value)  # numpy scalars and the like
        except (TypeError, ValueError):
            raise AnimationError(f"{name} must be a finite number, got {value!r}") from None
        if not math.isfinite(value):
            raise AnimationError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _vec3(value: object, name: str) -> tuple[float, float, float]:
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError:
        raise AnimationError(f"{name} must be three numbers, got {value!r}") from None
    if len(items) != 3:
        raise AnimationError(f"{name} must be three numbers, got {value!r}")
    return tuple(_number(item, name) for item in items)  # type: ignore[return-value]


def _unit(value: object, name: str) -> tuple[float, float, float]:
    x, y, z = _vec3(value, name)
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        raise AnimationError(f"{name} must be a nonzero vector, got {value!r}")
    return (x / n, y / n, z / n)


def _rigid(matrix: object) -> tuple:
    try:
        rows = [list(row) for row in matrix]  # type: ignore[union-attr]
    except TypeError:
        raise AnimationError(f"transform needs a 4x4 matrix, got {matrix!r}") from None
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise AnimationError("transform needs a 4x4 matrix (four rows of four)")
    m = [[_number(value, "transform matrix entry") for value in row] for row in rows]
    if max(abs(m[3][0]), abs(m[3][1]), abs(m[3][2]), abs(m[3][3] - 1.0)) > 1e-9:
        raise AnimationError("transform matrix's last row must be 0, 0, 0, 1")
    r = (m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2])
    for i in range(3):
        for j in range(3):
            dot = r[i * 3] * r[j * 3] + r[i * 3 + 1] * r[j * 3 + 1] + r[i * 3 + 2] * r[j * 3 + 2]
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-6:
                raise AnimationError("transform matrix must be rigid: its rotation is not orthonormal (no scale or shear)")
    det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6])
    if det < 0.0:
        raise AnimationError("transform matrix must be rigid: it mirrors")
    return r + (m[0][3], m[1][3], m[2][3])


# --- What a clip targets ---------------------------------------------------------


def _natural(occurrence_id: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in occurrence_id.lstrip("o").split("."))


class AnimationTargets:
    """The names and ids a clip's ``m.get`` resolves, each to document leaves."""

    def __init__(self, by_id: Mapping[str, Sequence[str]], by_name: Mapping[str, Sequence[str]]):
        self._by_id = {key: tuple(value) for key, value in by_id.items()}
        self._by_name = {key: tuple(value) for key, value in by_name.items()}
        self._resolved: dict[str, tuple[str, ...]] = {}

    def labels(self) -> list[str]:
        return sorted(self._by_name)

    def resolve(self, target: object) -> tuple[str, ...]:
        key = target if isinstance(target, str) else None
        cached = self._resolved.get(key) if key is not None else None
        if cached is not None:
            return cached
        if not isinstance(target, str) or not target.startswith("#") or not target[1:].strip():
            raise AnimationError(f"animation target {target!r} must be a #name or an #o1.2 occurrence id")
        selector = target[1:].strip()
        if selector in self._by_id:
            leaves = self._by_id[selector]
        else:
            leaves = tuple(dict.fromkeys(
                leaf for node in self._by_name.get(selector, ()) for leaf in self._by_id.get(node, ())
            ))
        if not leaves:
            known = ", ".join(f"#{name}" for name in self.labels()[:20]) or "(none)"
            more = " ..." if len(self._by_name) > 20 else ""
            raise AnimationError(f"animation target {target!r} names no part or group; names: {known}{more}")
        self._resolved[target] = leaves
        return leaves


def animation_targets(descriptor: Mapping[str, Any]) -> AnimationTargets:
    """Targets over a descriptor (an ``assembly.json``): its node names and ids,
    each resolving to the leaf occurrences beneath it."""
    from cadgen._internal.source_sidecar import _descriptor_nodes

    by_id, by_name = _descriptor_nodes(descriptor)
    return AnimationTargets(by_id, by_name)


# --- One sample ----------------------------------------------------------------


class _Frame:
    __slots__ = ("transform", "opacity", "visible")

    def __init__(self) -> None:
        self.transform: dict[str, tuple] = {}
        self.opacity: dict[str, float] = {}
        self.visible: dict[str, bool] = {}


class Handle:
    """The parts one ``m.get`` names. Every method returns the handle."""

    __slots__ = ("_frame", "_leaves")

    def __init__(self, frame: _Frame, leaves: tuple[str, ...]) -> None:
        self._frame = frame
        self._leaves = leaves

    def _apply(self, op: tuple) -> "Handle":
        transforms = self._frame.transform
        for leaf in self._leaves:
            current = transforms.get(leaf)
            transforms[leaf] = op if current is None else _compose(op, current)
        return self

    def rotate(self, axis: object, degrees: object, origin: object = (0.0, 0.0, 0.0)) -> "Handle":
        return self._apply(_rotation(_unit(axis, "rotate axis"), _number(degrees, "rotate degrees"), _vec3(origin, "rotate origin")))

    def translate(self, vector: object) -> "Handle":
        return self._apply(_IDENTITY[:9] + _vec3(vector, "translate vector"))

    def transform(self, matrix: object) -> "Handle":
        return self._apply(_rigid(matrix))

    def opacity(self, value: object) -> "Handle":
        value = min(1.0, max(0.0, _number(value, "opacity")))
        for leaf in self._leaves:
            self._frame.opacity[leaf] = value
        return self

    def visible(self, flag: object) -> "Handle":
        for leaf in self._leaves:
            self._frame.visible[leaf] = bool(flag)
        return self


class Model:
    """The ``m`` a clip's ``update(t, m)`` receives."""

    __slots__ = ("_frame", "_targets")

    def __init__(self, frame: _Frame, targets: AnimationTargets) -> None:
        self._frame = frame
        self._targets = targets

    def get(self, *targets: str) -> Handle:
        if not targets:
            raise AnimationError("m.get needs at least one #name or #occurrence id")
        if len(targets) == 1:
            return Handle(self._frame, self._targets.resolve(targets[0]))
        leaves = dict.fromkeys(leaf for target in targets for leaf in self._targets.resolve(target))
        return Handle(self._frame, tuple(leaves))

    def labels(self) -> list[str]:
        return self._targets.labels()


# --- Keyframe reduction --------------------------------------------------------


def _keep(count: int, error, tolerance: float, forced: Iterable[int] = (), reach=None) -> list[int]:
    """The sample indices to keep: both ends, every forced index, and enough
    between them that interpolating the kept samples rebuilds each dropped one
    within ``tolerance`` (``error(i, j, k)``: sample k's error between keys i, j).
    ``reach(i, j)``, when given, is the farthest sample before ``j`` that key
    ``i`` may span to, or None when it may span to ``j`` itself."""
    if count <= 2:
        return list(range(count))
    keep = {0, count - 1, *forced}
    anchors = sorted(keep)
    stack = list(zip(anchors, anchors[1:]))
    while stack:
        i, j = stack.pop()
        split = reach(i, j) if reach is not None else None
        if split is None:
            worst, split = tolerance, None
            for k in range(i + 1, j):
                e = error(i, j, k)
                if e > worst:
                    worst, split = e, k
        if split is not None:
            keep.add(split)
            stack += [(i, split), (split, j)]
    return sorted(keep)


def _corners(bounds: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    (x0, y0, z0), (x1, y1, z1) = bounds
    return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


def _apply_point(r: Sequence[float], t: Sequence[float], p: Sequence[float]) -> tuple[float, float, float]:
    return (
        r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
        r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
        r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2],
    )


def _solve3(a: list[list[float]], b: list[float]) -> list[float]:
    """``a x = b`` by Gaussian elimination with partial pivoting."""
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        x[r] = (m[r][3] - sum(m[r][c] * x[c] for c in range(r + 1, 3))) / m[r][r]
    return x


def _pivot(values: list[tuple], center: Sequence[float]) -> tuple[float, float, float]:
    """The point the motion moves least: argmin over c of the sum of |M_k(c) - c|^2.

    A part turning about a fixed axis leaves that axis still, so its keys need
    only the turn. Where the minimum is not one point -- along a fixed axis, or
    anywhere for a pure translation -- the tie goes to the point nearest the
    model's center."""
    a = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for v in values:
        d = ((v[0] - 1.0, v[1], v[2]), (v[3], v[4] - 1.0, v[5]), (v[6], v[7], v[8] - 1.0))
        for i in range(3):
            for j in range(3):
                a[i][j] += d[0][i] * d[0][j] + d[1][i] * d[1][j] + d[2][i] * d[2][j]
            b[i] -= d[0][i] * v[9] + d[1][i] * v[10] + d[2][i] * v[11]
    tie = 1e-9 * (a[0][0] + a[1][1] + a[2][2]) + 1e-30
    for i in range(3):
        a[i][i] += tie
        b[i] += tie * center[i]
    return tuple(_solve3(a, b))  # type: ignore[return-value]


def _transform_keys(
    times: list[float], values: list[tuple], corners: list[tuple], center: Sequence[float], tolerance: float, where: str
) -> tuple[list[int], tuple[float, float, float], list[list[float]]]:
    quats: list[tuple] = []
    swept = [0.0]  # degrees turned from the first sample, summed sample by sample
    for index, value in enumerate(values):
        q = _quaternion(value)
        if quats:
            previous = quats[-1]
            if previous[0] * q[0] + previous[1] * q[1] + previous[2] * q[2] + previous[3] * q[3] < 0.0:
                q = (-q[0], -q[1], -q[2], -q[3])  # the same rotation, on the near side
            turn = _turn_deg(previous, q)
            if turn > MAX_SAMPLE_TURN_DEG + 1e-6:
                raise AnimationError(
                    f"{where} turns {turn:.0f} degrees between the samples at "
                    f"{times[index - 1]:g} s and {times[index]:g} s: raise the clip's fps"
                )
            swept.append(swept[-1] + turn)
        quats.append(q)
    pivot = tuple(_length(c) for c in _pivot(values, center))
    # Each sample's key: d (where the pivot goes: M(pivot) - pivot), q, d's rate
    # and the angular velocity -- central differences, one-sided at the ends.
    moves = [[m - p for m, p in zip(_apply_point(v, v[9:], pivot), pivot)] for v in values]
    last = len(values) - 1
    keys = []
    for k in range(len(values)):
        i, j = max(0, k - 1), min(last, k + 1)
        span = times[j] - times[i]
        rate = [(moves[j][n] - moves[i][n]) / span for n in range(3)]
        spin = [c / span for c in _turn_vector(quats[i], quats[j])]
        keys.append([*moves[k], *quats[k], *rate, *spin])
    truth = [[_apply_point(value, value[9:], corner) for corner in corners] for value in values]
    arms = [tuple(c - p for c, p in zip(corner, pivot)) for corner in corners]

    def error(i: int, j: int, k: int) -> float:
        span = times[j] - times[i]
        d, r = _pose_at(keys[i], keys[j], span, (times[k] - times[i]) / span)
        at = (pivot[0] + d[0], pivot[1] + d[1], pivot[2] + d[2])
        return max(math.dist(_apply_point(r, at, arm), true) for arm, true in zip(arms, truth[k]))

    def reach(i: int, j: int) -> int | None:
        # The SWEPT turn, not the net one: a whole revolution between two keys
        # nets nothing and must still be split.
        if swept[j] - swept[i] <= MAX_KEY_TURN_DEG:
            return None
        return bisect.bisect_right(swept, swept[i] + MAX_KEY_TURN_DEG, i + 1, j) - 1

    keep = _keep(len(values), error, tolerance, reach=reach)
    digits = (_LENGTH_DIGITS,) * 3 + (_QUATERNION_DIGITS,) * 4 + (_LENGTH_DIGITS,) * 3 + (_QUATERNION_DIGITS,) * 3
    return keep, pivot, [[_round(c, digits[n]) for n, c in enumerate(keys[k])] for k in keep]


def _runs(values: list[Any], kind) -> list[int]:
    """Indices that must stay keys because the value's kind changes there: the
    last sample of one kind and the first of the next."""
    forced: list[int] = []
    for k in range(1, len(values)):
        if kind(values[k]) != kind(values[k - 1]):
            forced += [k - 1, k]
    return forced


def _round(value: float, digits: int) -> float:
    return round(value, digits) + 0.0  # never -0.0


def _length(value: float) -> float:
    return _round(value, _LENGTH_DIGITS)


# --- A clip, end to end ----------------------------------------------------------


def sample_times(duration: float, fps: float) -> list[float]:
    """0, 1/fps, ... and ``duration`` itself: the last pose is always sampled."""
    count = max(1, math.ceil(duration * fps - 1e-9))
    return [index / fps for index in range(count)] + [duration]


def bake_clip(clip_id: str, clip: Any, targets: AnimationTargets, bounds: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Sample one :class:`cadgen.animation.Clip` and reduce it to tracks."""
    times = sample_times(clip.duration, clip.fps)
    frames: list[_Frame] = []
    for t in times:
        frame = _Frame()
        try:
            clip.update(t, Model(frame, targets))
        except AnimationError as exc:
            raise AnimationError(f"animation clip {clip_id!r} at t={t:g} s: {exc}") from None
        except Exception as exc:
            raise AnimationError(f"animation clip {clip_id!r} at t={t:g} s: {type(exc).__name__}: {exc}") from exc
        frames.append(frame)

    diagonal = math.dist(bounds[0], bounds[1])
    tolerance = max(LENGTH_FLOOR, TRANSFORM_TOLERANCE * diagonal)
    corners = _corners(bounds)
    center = tuple((lo + hi) / 2.0 for lo, hi in zip(bounds[0], bounds[1]))
    rounded_times = [_round(t, _TIME_DIGITS) for t in times]
    tracks: list[dict[str, Any]] = []

    def channel(name: str, rest: Any) -> dict[tuple, list[str]]:
        """Leaves grouped by their whole sequence on this channel: parts that move
        identically share a track. A leaf no sample touched is left out."""
        leaves = dict.fromkeys(leaf for frame in frames for leaf in getattr(frame, name))
        groups: dict[tuple, list[str]] = {}
        for leaf in leaves:
            sequence = tuple(getattr(frame, name).get(leaf, rest) for frame in frames)
            groups.setdefault(sequence, []).append(leaf)
        return groups

    for sequence, leaves in channel("transform", _IDENTITY).items():
        if all(value == sequence[0] for value in sequence) and _is_identity(sequence[0]):
            continue  # touched, never moved
        where = f"animation clip {clip_id!r} part {sorted(leaves, key=_natural)[0]}"
        keep, pivot, values = _transform_keys(times, list(sequence), corners, center, tolerance, where)
        track = _track(leaves, rounded_times, keep, "transform", values)
        track["pivot"] = list(pivot)
        tracks.append(track)

    for sequence, leaves in channel("opacity", None).items():
        values = list(sequence)

        def opacity_error(i: int, j: int, k: int) -> float:
            if values[i] is None:  # a run of nulls: _runs anchors both ends, so all null, held
                return 0.0
            u = (times[k] - times[i]) / (times[j] - times[i])
            return abs(values[i] + (values[j] - values[i]) * u - values[k])

        keep = _keep(len(values), opacity_error, OPACITY_TOLERANCE, _runs(values, lambda v: v is None))
        tracks.append(_track(leaves, rounded_times, keep, "opacity", [
            None if values[k] is None else _round(values[k], _OPACITY_DIGITS) for k in keep
        ]))

    for sequence, leaves in channel("visible", None).items():
        values = list(sequence)
        keep = [0] + [k for k in range(1, len(values)) if values[k] != values[k - 1]]
        tracks.append(_track(leaves, rounded_times, keep, "visible", [values[k] for k in keep]))

    tracks.sort(key=lambda track: (CHANNELS.index(_channel_of(track)), _natural(track["targets"][0])))
    return {
        "id": clip_id,
        "label": clip.label or clip_id,
        "duration": clip.duration,
        "loop": clip.loop,
        "tracks": tracks,
    }


def _track(leaves: list[str], times: list[float], keep: list[int], channel: str, values: list[Any]) -> dict[str, Any]:
    # A track whose every key is the same value is that value, held: one key.
    if len(values) > 1 and all(value == values[0] for value in values[1:]):
        keep, values = keep[:1], values[:1]
    return {"targets": sorted(leaves, key=_natural), "times": [times[k] for k in keep], channel: values}


def _channel_of(track: Mapping[str, Any]) -> str:
    return next(name for name in CHANNELS if name in track)


def bake_animation(clips: Mapping[str, Any], targets: AnimationTargets, bounds: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Every clip of a model, baked: the sidecar's ``animation`` section."""
    return {"clips": [bake_clip(clip_id, clip, targets, bounds) for clip_id, clip in clips.items()]}


def bake_document_animation(clips: Mapping[str, Any], document_tree: str) -> dict[str, Any]:
    """Bake against the WRITTEN document's tree: its names, its occurrence ids
    and its bounds are exactly what every renderer of the file holds."""
    from cadgen.store.trees import flatten

    descriptor = flatten(document_tree)
    if descriptor is None:
        raise AnimationError(f"cannot bake animation: document tree {document_tree} is missing")
    bbox = descriptor.get("bbox") or {}
    if not bbox.get("min") or not bbox.get("max"):
        raise AnimationError("cannot bake animation: the model has no geometry to measure")
    return bake_animation(clips, animation_targets(descriptor), (bbox["min"], bbox["max"]))


# --- The section, read back ------------------------------------------------------


def _fail(message: str) -> ValueError:
    return ValueError(f"animation: {message}")


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def normalize_baked_animation(block: object) -> dict[str, Any] | None:
    """Check a sidecar's ``animation`` section has the shape the renderers read.
    Returns the block (or ``None`` for an empty one); raises with what is wrong."""
    if block is None:
        return None
    if not isinstance(block, Mapping) or set(block) != {"clips"} or not isinstance(block["clips"], list):
        raise _fail("the section must be {'clips': [...]}")
    if not block["clips"]:
        return None
    seen: set[str] = set()
    for index, clip in enumerate(block["clips"]):
        if not isinstance(clip, Mapping) or set(clip) != {"id", "label", "duration", "loop", "tracks"}:
            raise _fail(f"clip {index} must have exactly id, label, duration, loop and tracks")
        clip_id = clip["id"]
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen:
            raise _fail(f"clip {index} needs an id of its own, got {clip_id!r}")
        seen.add(clip_id)
        where = f"clip {clip_id!r}"
        if not isinstance(clip["label"], str) or not clip["label"] or not _finite(clip["duration"]) or clip["duration"] <= 0 or not isinstance(clip["loop"], bool):
            raise _fail(f"{where} needs a label, a positive duration and a boolean loop")
        if not isinstance(clip["tracks"], list):
            raise _fail(f"{where} tracks must be a list")
        for index, track in enumerate(clip["tracks"]):
            _check_track(track, f"{where} track {index}", clip["duration"])
    return dict(block)


def _check_track(track: object, where: str, duration: float) -> None:
    if not isinstance(track, Mapping):
        raise _fail(f"{where} must be an object")
    channels = [name for name in CHANNELS if name in track]
    extra = {"transform": {"pivot"}}
    allowed = {"targets", "times", *channels} | (extra.get(channels[0], set()) if len(channels) == 1 else set())
    if len(channels) != 1 or set(track) - allowed:
        raise _fail(f"{where} must carry targets, times and exactly one of {', '.join(CHANNELS)}")
    targets, times, values = track.get("targets"), track.get("times"), track[channels[0]]
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) and t for t in targets):
        raise _fail(f"{where} targets must be a nonempty list of occurrence ids")
    if (
        not isinstance(times, list) or not times or not all(_finite(t) for t in times)
        or times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])) or times[-1] > duration + 1e-9
    ):
        raise _fail(f"{where} times must rise strictly from 0 to at most the duration")
    if not isinstance(values, list) or len(values) != len(times):
        raise _fail(f"{where} needs one {channels[0]} value per time")
    channel = channels[0]
    for value in values:
        if channel == "transform":
            ok = isinstance(value, list) and len(value) == 13 and all(_finite(c) for c in value)
        elif channel == "opacity":
            ok = value is None or (_finite(value) and 0 <= value <= 1)
        else:
            ok = value is None or isinstance(value, bool)
        if not ok:
            raise _fail(f"{where} has a malformed {channel} value: {value!r}")
    if channel == "transform":
        pivot = track.get("pivot")
        if not isinstance(pivot, list) or len(pivot) != 3 or not all(_finite(c) for c in pivot):
            raise _fail(f"{where} needs its pivot, three numbers")
