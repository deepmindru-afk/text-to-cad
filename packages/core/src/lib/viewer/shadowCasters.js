// What a scene's display records put into a shadow pass, as a snapshot a later pass compares
// against. A shadow map shows only which records cast, where each one is and what shape it
// has; a pass over the records that changed none of that (a hover, a selection, a highlight
// colour, a pose pass that re-ran without moving anything) leaves every shadow map, and the
// floor shadow baked from them, exactly as it is, and need not render them again.
//
// Per record: the record itself, whether it casts (visible and `castShadow`), its local
// matrix, which the pose, the animation and the exploded view all write, and its geometry.
// Nothing else of a record reaches a shadow pass: colour, emission, opacity, render order and
// draw membership (instanced or not) never change what a caster occludes.
//
// A playing routine takes one snapshot and one comparison a frame, so the comparison reads
// the records in place and stops at the first difference.

const VALUES_PER_RECORD = 17;

function casts(mesh) {
  return mesh && mesh.visible !== false && mesh.castShadow === true ? 1 : 0;
}

/**
 * @param {Array<object>} records  display records: `{ mesh, ... }`
 * @returns {{ records: object[], values: Float64Array, geometries: Array<unknown> }}
 */
export function captureShadowCasters(records) {
  const list = Array.isArray(records) ? records.slice() : [];
  const values = new Float64Array(list.length * VALUES_PER_RECORD);
  const geometries = new Array(list.length);
  for (let index = 0; index < list.length; index += 1) {
    const mesh = list[index]?.mesh;
    const offset = index * VALUES_PER_RECORD;
    values[offset] = casts(mesh);
    const elements = mesh?.matrix?.elements;
    if (elements) for (let element = 0; element < 16; element += 1) values[offset + 1 + element] = elements[element];
    geometries[index] = mesh?.geometry ?? null;
  }
  return { records: list, values, geometries };
}

/**
 * Whether `records` would cast a different shadow than when `before` was captured. Without a
 * snapshot to compare against, they might.
 */
export function shadowCastersChanged(before, records) {
  if (!before) return true;
  const list = Array.isArray(records) ? records : [];
  if (list.length !== before.records.length) return true;
  const { values, geometries } = before;
  for (let index = 0; index < list.length; index += 1) {
    const record = list[index];
    if (record !== before.records[index]) return true;
    const mesh = record?.mesh;
    const offset = index * VALUES_PER_RECORD;
    if (casts(mesh) !== values[offset]) return true;
    const elements = mesh?.matrix?.elements;
    if (elements) {
      for (let element = 0; element < 16; element += 1) {
        if (!Object.is(elements[element], values[offset + 1 + element])) return true;
      }
    }
    if ((mesh?.geometry ?? null) !== geometries[index]) return true;
  }
  return false;
}
