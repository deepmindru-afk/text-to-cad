// The CI gate for the hero showcase assets: the committed render package and
// sidecar under public/hero/ must still satisfy the contracts the hero page
// consumes through @text-to-cad/core — each component's mesh, as cadgen made
// it, readable as that component's mesh at the tolerances the hero draws; a
// kinematics section that compiles into a step-module definition; and embedded
// animation source. A cadgen format bump that regenerates these fails here
// instead of silently breaking the production render. Refresh with
// scripts/sync-hero-step-assets.mjs.

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { packageSourceFromBaseUrl } from "@text-to-cad/core/common/source.js";
import { decodeComponentTessellation } from "@text-to-cad/core/lib/surf/tessellationCache.js";
import { stepModuleFromKinematics } from "@text-to-cad/core/common/kinematicsModule.js";
import { loadSourceAnimation } from "@text-to-cad/core/common/animationRuntime.js";
import { validateSourceSidecar } from "@text-to-cad/core/common/sourceSidecar.js";

const docsRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const heroPackageDir = path.join(docsRoot, "public/hero/planetary");
const heroSidecarPath = path.join(docsRoot, "public/hero/planetary_gear_assembly.step.json");

const descriptor = JSON.parse(
  fs.readFileSync(path.join(heroPackageDir, "assembly.json"), "utf8"),
);
const components = Object.entries(descriptor.components || {});
assert.ok(components.length > 0, "Hero render package descriptor lists no components");

// The mesh URLs the hero reads, resolved the way the page resolves them.
const { package: heroPackage } = packageSourceFromBaseUrl("/hero/planetary", descriptor);
for (const [cid, entry] of components) {
  const meshPath = path.join(docsRoot, "public", heroPackage.meshUrls[cid]);
  assert.ok(fs.existsSync(meshPath), `Hero component ${cid} is missing its mesh (${meshPath})`);
  const decoded = decodeComponentTessellation(new Uint8Array(fs.readFileSync(meshPath)), {
    surfaceInput: entry.surfaceInput, surfaceObject: entry.surfaceObject, tessellation: {},
  });
  assert.ok(decoded, `${meshPath} is not component ${cid}'s mesh at the default tolerances in the current TESS format`);
  assert.ok(decoded.component.indices.length > 0, `${meshPath} draws no triangles`);
}

const rawSidecar = JSON.parse(fs.readFileSync(heroSidecarPath, "utf8"));
const sidecar = validateSourceSidecar(rawSidecar, {
  url: heroSidecarPath,
  documentHash: rawSidecar.documentHash,
});
assert.ok(
  stepModuleFromKinematics(sidecar.kinematics),
  "Hero sidecar kinematics no longer compile into a step-module definition",
);
const animation = await loadSourceAnimation(sidecar);
assert.ok(animation?.clips?.meshCycle, "Hero sidecar does not declare the meshCycle clip");

console.log(`Hero STEP assets are current: ${components.length} components, schema-v10 kinematics + clips OK.`);
