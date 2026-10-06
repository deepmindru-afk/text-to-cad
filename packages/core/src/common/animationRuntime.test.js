import assert from "node:assert/strict";
import { test } from "node:test";
import * as THREE from "three";

import {
  applyAnimationFrameToEffects,
  evaluateAnimationClip,
  isAnimationClip,
  loadSourceAnimation,
  normalizeAnimationClips
} from "./animationRuntime.js";
import { animationClipList } from "./animationClock.js";

// Keyframes written by hand, the way cadgen bakes them: a transform key is
// [d, q, d', w], the track's pivot moved by d while the part turns by q about it.
// An oblique axis, so every term of the quaternion's matrix is exercised.
const AXIS = new THREE.Vector3(1, 2, 2).normalize();
const PIVOT = [10, 0, 0];
const radians = THREE.MathUtils.degToRad;

/** A transform key: the pivot moved by `d` at `rate`, turned `deg` about AXIS at `degPerSec`. */
function key(d, deg, rate = [0, 0, 0], degPerSec = 0) {
  const q = new THREE.Quaternion().setFromAxisAngle(AXIS, radians(deg));
  const w = AXIS.clone().multiplyScalar(radians(degPerSec));
  return [...d, q.x, q.y, q.z, q.w, ...rate, w.x, w.y, w.z];
}

/** The placement a key names, T(pivot + d) R T(-pivot), composed by three. */
function pose(d, deg) {
  return new THREE.Matrix4().makeTranslation(PIVOT[0] + d[0], PIVOT[1] + d[1], PIVOT[2] + d[2])
    .multiply(new THREE.Matrix4().makeRotationAxis(AXIS, radians(deg)))
    .multiply(new THREE.Matrix4().makeTranslation(-PIVOT[0], -PIVOT[1], -PIVOT[2]));
}

function assertPose(matrix, expected, label) {
  matrix.elements.forEach((value, index) => {
    assert.ok(Math.abs(value - expected.elements[index]) < 1e-9, `${label}: element ${index} is ${value}, not ${expected.elements[index]}`);
  });
}

const clip = (tracks, { duration = 4, loop = true } = {}) => ({ id: "demo", label: "Demo", duration, loop, tracks });
const transformTrack = (times, keys) => ({ targets: ["o1.2", "o1.3"], times, pivot: PIVOT, transform: keys });
const at = (animation, t) => evaluateAnimationClip(THREE, animation, t);

test("clips normalize by id in the order the model declares them", () => {
  const clips = normalizeAnimationClips({ clips: [
    { id: "zoom", label: "Zoom", duration: 2, loop: false, tracks: [] },
    { id: "approach", label: "Approach", duration: 1, loop: true, tracks: [] }
  ] });
  assert.deepEqual(Object.keys(clips), ["zoom", "approach"]);
  assert.deepEqual(clips.zoom, { id: "zoom", label: "Zoom", duration: 2, loop: false, tracks: [], order: 0 });
  assert.equal(isAnimationClip(clips.approach), true);
  // The retired module form, a clip with an update function, is not a clip.
  assert.equal(isAnimationClip({ duration: 1, update() {} }), false);
  assert.equal(isAnimationClip(null), false);
});

test("the clip list keeps the declared order, even for ids an object would reorder", () => {
  const clips = normalizeAnimationClips({ clips: [
    { id: "2", label: "Second", duration: 1, loop: true, tracks: [] },
    { id: "1", label: "First", duration: 1, loop: true, tracks: [] }
  ] });
  // An object lists integer-like keys first; the viewer must still open on "2".
  assert.deepEqual(Object.keys(clips), ["1", "2"]);
  assert.deepEqual(animationClipList(clips).map((clip) => clip.id), ["2", "1"]);
});

test("a sidecar's animation loads validated, and a sidecar with none loads as null", async () => {
  for (const sidecar of [{}, { animation: null }, { animation: { clips: [] } }]) {
    assert.equal(await loadSourceAnimation(sidecar), null);
  }
  const swing = { id: "swing", label: "Swing", duration: 4, loop: true, tracks: [transformTrack([0], [key([0, 0, 0], 0)])] };
  const { clips } = await loadSourceAnimation({ animation: { clips: [swing] } });
  assert.deepEqual(clips, { swing: { ...swing, order: 0 } });
  await assert.rejects(loadSourceAnimation({ animation: { clips: { swing } } }), /animation: the section must be/);
  await assert.rejects(loadSourceAnimation({ animation: { clips: [swing] } }, { signal: AbortSignal.abort() }), { name: "AbortError" });
});

test("a constant spin about an off-origin pivot is exact between keys", () => {
  // 120 deg/s about AXIS through PIVOT while the pivot rises 5 mm/s, keyed one second
  // apart: every moment between the keys is the motion itself, not an approximation.
  const screw = clip([transformTrack([0, 1], [
    key([0, 0, 0], 0, [0, 0, 5], 120),
    key([0, 0, 5], 120, [0, 0, 5], 120)
  ])]);
  for (const t of [0.25, 0.5, 0.8]) {
    const { matrices } = at(screw, t);
    assertPose(matrices.get("o1.2"), pose([0, 0, 5 * t], 120 * t), `t=${t}`);
    assert.deepEqual(matrices.get("o1.3").elements, matrices.get("o1.2").elements, "a track moves its targets alike");
  }
});

test("a key at its own time is exact, and a track holds its end keys outside its times", () => {
  // The interior key's rates disagree with its neighbours: at its own time it is still itself.
  const track = transformTrack([0, 1, 2], [key([0, 0, 0], 0), key([1, 2, 3], 45, [9, 9, 9], 300), key([4, 0, 0], 90)]);
  const swing = clip([track]);
  assertPose(at(swing, 0).matrices.get("o1.2"), pose([0, 0, 0], 0), "first key");
  assertPose(at(swing, 1).matrices.get("o1.2"), pose([1, 2, 3], 45), "interior key");
  assertPose(at(swing, 2).matrices.get("o1.2"), pose([4, 0, 0], 90), "last key");
  assertPose(at(swing, 3.5).matrices.get("o1.2"), pose([4, 0, 0], 90), "after the last key");
  assertPose(at(swing, -1).matrices.get("o1.2"), pose([0, 0, 0], 0), "before the first key");
  // A track of one key holds it for the whole clip.
  const held = clip([transformTrack([0], [key([0, 0, 7], 30)])]);
  for (const t of [0, 1.7, 3.99]) assertPose(at(held, t).matrices.get("o1.2"), pose([0, 0, 7], 30), `one key, t=${t}`);
});

test("a looping clip wraps its time, and one that does not clamps it", () => {
  const slide = [transformTrack([0, 2], [key([0, 0, 0], 0, [1, 0, 0]), key([2, 0, 0], 0, [1, 0, 0])])];
  assertPose(at(clip(slide, { duration: 2 }), 2.5).matrices.get("o1.2"), pose([0.5, 0, 0], 0), "looping");
  assertPose(at(clip(slide, { duration: 2, loop: false }), 99).matrices.get("o1.2"), pose([2, 0, 0], 0), "clamped");
});

test("opacity lerps between numbers and holds against null; visibility holds", () => {
  const fade = clip([
    { targets: ["o1.2"], times: [0, 1, 2, 3], opacity: [null, 0.25, 0.75, null] },
    { targets: ["o1.3"], times: [0, 1, 2], visible: [false, true, null] }
  ]);
  const styles = (t) => Object.fromEntries(at(fade, t).styles);
  // null is the material's own opacity and the part's rest visibility: no style at all.
  assert.deepEqual(styles(0.5), { "o1.3": { visible: false } });
  assert.deepEqual(styles(1.5), { "o1.2": { opacity: 0.5 }, "o1.3": { visible: true } });
  // A number before a null holds rather than fading toward the material's own.
  assert.deepEqual(styles(2.5), { "o1.2": { opacity: 0.75 } });
  assert.deepEqual(styles(3.5), {});
});

function through(matrix, point) {
  return new THREE.Vector3(...point).applyMatrix4(matrix).toArray().map((v) => Math.round(v * 1e6) / 1e6);
}

const frame = ({ matrices = [], styles = [] } = {}) => ({ matrices: new Map(matrices), styles: new Map(styles) });

test("frame effects premultiply onto an existing pose matrix", () => {
  // Pose put the part at x=10; the clip then orbits the origin by 90deg. The
  // animation must act on the ALREADY-POSED part (world space), not under it.
  const effectsByPartId = new Map([
    ["o1.1", {
      matrix: new THREE.Matrix4().makeTranslation(10, 0, 0),
      style: null,
      visible: null,
      highlighted: false
    }]
  ]);
  const orbit = frame({ matrices: [["o1.1", new THREE.Matrix4().makeRotationZ(Math.PI / 2)]] });
  const transformCount = applyAnimationFrameToEffects(THREE, effectsByPartId, orbit);
  assert.equal(transformCount, 1);
  assert.deepEqual(through(effectsByPartId.get("o1.1").matrix, [0, 0, 0]), [0, 10, 0]);
});

test("frame effects open records for untouched parts and merge styles", () => {
  const effectsByPartId = new Map();
  const styled = frame({ styles: [["o1.1", { opacity: 0.5 }], ["o1.2", { visible: false }]] });
  assert.equal(applyAnimationFrameToEffects(THREE, effectsByPartId, styled), 0);
  assert.deepEqual(effectsByPartId.get("o1.1").style, { opacity: 0.5 });
  assert.equal(effectsByPartId.get("o1.2").visible, false);
  assert.equal(effectsByPartId.get("o1.1").matrix, null);
});
