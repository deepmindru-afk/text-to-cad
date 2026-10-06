import assert from "node:assert/strict";
import test from "node:test";

import * as THREE from "three";

import { captureShadowCasters, shadowCastersChanged } from "./shadowCasters.js";

function record(partId) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial());
  mesh.castShadow = true;
  mesh.matrixAutoUpdate = false;
  return { partId, mesh, material: mesh.material };
}

test("a pass that only recolours, highlights or re-renders its records changes no shadow", () => {
  const records = [record("a"), record("b")];
  const before = captureShadowCasters(records);
  // What a hover or a selection writes: colour, emission, transparency, opacity, render order.
  records[0].material.color.set("#ff00ff");
  records[0].material.emissive.set("#ff00ff");
  records[0].material.transparent = true;
  records[0].material.opacity = 0.5;
  records[0].mesh.renderOrder = 23;
  records[1].mesh.receiveShadow = false;
  // A pose pass that wrote the same matrix again.
  records[1].mesh.matrix.copy(records[1].mesh.matrix.clone());
  assert.equal(shadowCastersChanged(before, records), false);
});

test("moving, hiding, or taking a record out of the shadow pass is a change", () => {
  assert.equal(shadowCastersChanged(null, [record("a")]), true, "nothing to compare against");
  const cases = [
    (records) => records[0].mesh.matrix.makeTranslation(0, 0, 1e-9),
    (records) => { records[1].mesh.visible = false; },
    (records) => { records[1].mesh.castShadow = false; },
    (records) => { records[1].mesh.geometry = new THREE.SphereGeometry(1); },
    (records) => records.pop(),
    (records) => records.splice(0, 1, record("a")),
    // A rebuilt record is not vouched for, even on the same mesh.
    (records) => records.splice(0, 1, { ...records[0] }),
  ];
  for (const change of cases) {
    const records = [record("a"), record("b")];
    const before = captureShadowCasters(records);
    change(records);
    assert.equal(shadowCastersChanged(before, records), true, String(change));
  }
});
