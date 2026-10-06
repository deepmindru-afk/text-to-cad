// The runtime draws, between keys, exactly what cadgen's baker assumed it would.
//
// cadgen drops a key only when ITS interpolation rebuilds the dropped samples
// within tolerance (cadgen/_internal/animation_bake.py), so the guarantee holds
// only while this runtime interpolates identically. animationRuntime.parity.json
// holds baked keyframes and the frames a renderer must draw from them at times
// between keys; cadgen's test_animation_parity.py pins the Python half against
// the same file.

import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";
import * as THREE from "three";

import { evaluateAnimationClip, loadSourceAnimation } from "./animationRuntime.js";

const parity = JSON.parse(fs.readFileSync(new URL("./animationRuntime.parity.json", import.meta.url), "utf8"));

function assertClose(got, want, where) {
  if (Array.isArray(want)) {
    assert.equal(got.length, want.length, where);
    want.forEach((value, index) => assertClose(got[index], value, `${where}[${index}]`));
  } else if (want && typeof want === "object") {
    assert.deepEqual(Object.keys(got).sort(), Object.keys(want).sort(), where);
    for (const key of Object.keys(want)) assertClose(got[key], want[key], `${where}.${key}`);
  } else if (typeof want === "number") {
    assert.ok(Math.abs(got - want) <= 1e-9, `${where}: ${got} != ${want}`);
  } else {
    assert.equal(got, want, where);
  }
}

test("every probe frame is what cadgen's baker interpolates, on every channel", async () => {
  const { clips } = await loadSourceAnimation({ animation: parity.animation });
  for (const probe of parity.probes) {
    const where = `${probe.clip}@${probe.t.toFixed(4)}`;
    const frame = evaluateAnimationClip(THREE, clips[probe.clip], probe.t);
    const matrices = Object.fromEntries([...frame.matrices].map(([id, matrix]) => {
      const e = matrix.elements; // column-major
      return [id, [0, 1, 2, 3].map((row) => [0, 1, 2, 3].map((column) => e[column * 4 + row]))];
    }));
    assertClose(matrices, probe.matrices, `${where}.matrices`);
    assertClose(Object.fromEntries(frame.styles), probe.styles, `${where}.styles`);
  }
});
