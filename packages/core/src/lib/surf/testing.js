// TEST-ONLY: synthetic TESS v5 bodies, for tests in any package that need a
// stored mesh no producer made (a single triangle, an empty or wire-only
// component, hundreds of distinct components). cadgen is the only producer of
// real meshes (cadgen/store/meshes.py encode_payload); nothing in the product
// imports this module. Browser-safe, so browser tests can use it too.

import {
  TESS_CACHE_MAGIC,
  TESS_CACHE_VERSION,
  TESSELLATION_VERSION,
  tessellationCacheKey,
  tessellationQuality,
} from "./tessellationCache.js";

/**
 * A synthetic TESS v5 body for `component` ({positions, normals, faceOrds, indices, faceRanges,
 * edges: [{ord, visibilityClass, polyline}], bounds, scale}), laid out as cadgen writes one.
 */
export function encodeTessFixture(component, {
  surfaceInput,
  surfaceObject,
  tessellation = {},
  partColor = null,
  edgeClasses,
} = {}) {
  const edges = Array.isArray(component.edges) ? component.edges : [];
  const quality = tessellationQuality(tessellation);
  const header = {
    tessellationInput: tessellationCacheKey(surfaceInput, tessellation),
    surfaceInput,
    surfaceDigest: surfaceObject,
    quality,
    tessellatorVersion: TESSELLATION_VERSION,
    payloadVersion: TESS_CACHE_VERSION,
    partColor: partColor ?? null,
    edgeClasses: edgeClasses ?? edges.map((edge) => [edge.ord, edge.visibilityClass ?? "none"]),
    faceRanges: component.faceRanges,
    bounds: { min: [...component.bounds.min], max: [...component.bounds.max] },
    scale: component.scale,
    positionCount: component.positions.length,
    normalCount: component.normals.length,
    faceOrdCount: component.faceOrds.length,
    indexCount: component.indices.length,
    edges: edges.map((edge) => ({
      ord: edge.ord, visibilityClass: edge.visibilityClass ?? null, count: edge.polyline.length,
    })),
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const headerLength = (headerBytes.length + 3) & ~3;
  const arrays = [
    [component.positions, Float32Array], [component.normals, Float32Array], [component.faceOrds, Float32Array],
    [component.indices, Uint32Array], ...edges.map((edge) => [edge.polyline, Float32Array]),
  ];
  const bytes = new Uint8Array(12 + headerLength + arrays.reduce((sum, [array]) => sum + array.length * 4, 0));
  const view = new DataView(bytes.buffer);
  view.setUint32(0, TESS_CACHE_MAGIC, true);
  view.setUint32(4, TESS_CACHE_VERSION, true);
  view.setUint32(8, headerLength, true);
  bytes.set(headerBytes, 12);
  bytes.fill(0x20, 12 + headerBytes.length, 12 + headerLength);
  let offset = 12 + headerLength;
  for (const [array, Ctor] of arrays) {
    new Ctor(bytes.buffer, offset, array.length).set(array);
    offset += array.length * 4;
  }
  return bytes;
}
