import { readCadWorkerTicket } from "../../client/resources.js";
// Surf component worker.
//
// One request = one component's stored mesh (bytes the client thread read and
// verified) and an explicit capability set. Ordinary display asks only for
// render data, which the mesh alone carries; picking/measurement asks for
// selectors, which also read the component's .surf for its topology tables;
// refinement of active topology asks for both so triangle ranges stay aligned.
// Decoding, copying into render-owned arrays and building selectors run here,
// off the page's main thread.

import { parseSurf } from "./container.js";
import { buildMeshDataFromSurf } from "./surfMeshData.js";
import { buildSelectorBundleFromSurf } from "./surfSelectorBundle.js";
import { meshDataTransferList } from "../render/meshTransfer.js";
import { decodeComponentTessellation, surfIndexFromCacheEntry } from "./tessellationCache.js";

const activeControllers = new Map();

function bundleTransferList(bundle) {
  return Object.values(bundle?.buffers || {})
    .map((view) => view?.buffer)
    .filter((buffer) => buffer instanceof ArrayBuffer && buffer.byteLength > 0);
}

function requestedCapabilities(message) {
  const value = message?.capabilities;
  if (!value || typeof value !== "object") {
    return { render: true, selectors: true };
  }
  return {
    render: value.render === true,
    selectors: value.selectors === true,
  };
}

self.addEventListener("message", async (event) => {
  const message = event.data || {};
  const id = message.id;
  if (!id) {
    return;
  }
  if (message.type === "cancel") {
    activeControllers.get(id)?.abort();
    activeControllers.delete(id);
    return;
  }
  if (message.type !== "loadSurf") {
    return;
  }
  const controller = new AbortController();
  activeControllers.set(id, controller);
  try {
    const capabilities = requestedCapabilities(message);
    if (!capabilities.render && !capabilities.selectors) {
      throw new Error("Surf worker request has no capabilities");
    }
    const cacheIdentity = message.cacheIdentity || {};
    const cached = decodeComponentTessellation(message.cachedEntry, {
      surfaceInput: cacheIdentity.surfaceInput,
      surfaceObject: cacheIdentity.surfaceObject,
      tessellation: message.tessellation || {},
    });
    if (!cached) {
      throw new Error("Surf worker request carries no readable mesh for this component");
    }
    let index = surfIndexFromCacheEntry(cached);
    if (capabilities.selectors) {
      const buffer = await readCadWorkerTicket(message.resource, { signal: controller.signal });
      ({ index } = parseSurf(buffer));
    }
    const meshData = capabilities.render ? buildMeshDataFromSurf(index, cached.component) : null;
    const bundle = capabilities.selectors ? buildSelectorBundleFromSurf(index, cached.component) : null;
    if (controller.signal.aborted) {
      return;
    }
    self.postMessage(
      {
        id,
        ok: true,
        ...(meshData ? { meshData } : {}),
        ...(bundle ? { bundle } : {}),
      },
      [...new Set([
        ...(meshData ? meshDataTransferList(meshData) : []),
        ...bundleTransferList(bundle),
      ])],
    );
  } catch (error) {
    if (!controller.signal.aborted) {
      self.postMessage({
        id,
        ok: false,
        error: {
          name: error?.name || "Error",
          message: error instanceof Error ? error.message : String(error),
        },
      });
    }
  } finally {
    activeControllers.delete(id);
  }
});
