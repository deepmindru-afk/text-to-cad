"""One component's display mesh: OCCT's mesher on the exact BREP, as TESS v5.

cadgen's only tessellator. ``BRepMesh_IncrementalMesh`` -- the mesher build123d's
own exporters use -- triangulates the component's exact faces at a chord
tolerance RELATIVE to the component's bounding diagonal and an angular one in
radians, discretizing every edge once so the faces either side share its
vertices. What it writes is one TESS body (``cadgen.store.meshes``): triangles
grouped per face, smooth normals from the exact surfaces, and one polyline per
model edge lying exactly on the mesh boundary. The viewer, snapshots and mesh
exports all draw these same bytes.

Face and edge ordinals are ``TopExp.MapShapes`` order on the unlocated
component -- the order its SURF index and every selector use -- and the SURF
index supplies what the triangles do not: each face's intrinsic colour and
each edge's display class. A face with area that does not mesh is an error,
never a hole (README law 10).

The arrays leave OCCT through its own glTF writer (``RWGltf_CafWriter``, one
primitive per face), not one Python call per vertex: that is what keeps a large
component's extraction in milliseconds. A component whose faces cannot be
matched to the writer's primitives one for one (a face its explorer meets
twice, for instance) is read face by face instead, which is slower and the same.
"""

from __future__ import annotations

import json
import math
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Two points closer than this, relative to the component's diagonal, are one
# point: the floor under the scale of a component with no extent.
_SCALE_FLOOR = 1e-6
# A face whose exact area is below this fraction of the diagonal squared may
# legitimately produce no triangles; anything larger must mesh.
_NEGLIGIBLE_AREA = 1e-12


class MeshProductionError(ValueError):
    """A component OCCT could not mesh completely."""


def _bounding_diagonal(topods) -> float:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(topods, box, False)
    if box.IsVoid():
        return _SCALE_FLOOR
    x0, y0, z0, x1, y1, z1 = box.Get()
    diagonal = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    return diagonal if math.isfinite(diagonal) and diagonal > _SCALE_FLOOR else _SCALE_FLOOR


def _maps(topods):
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    faces, edges = TopTools_IndexedMapOfShape(), TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(topods, TopAbs_FACE, faces)
    TopExp.MapShapes_s(topods, TopAbs_EDGE, edges)
    return faces, edges


def _triangulated_faces(topods, face_map) -> list[tuple[int, Any, Any, Any]]:
    """(ordinal, face, triangulation, location) in explorer order, as the glTF writer meets them."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    found = []
    explorer = TopExp_Explorer(topods, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None and triangulation.NbTriangles() > 0:
            found.append((face_map.FindIndex(face), face, triangulation, location))
        explorer.Next()
    return found


def _faces_from_gltf(topods, triangulated) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """Each face's (positions, normals, local triangles), written by OCCT's glTF writer.

    None when the writer's primitives cannot be matched to ``triangulated`` one
    for one; the caller then reads the faces itself."""
    from OCP.Message import Message_ProgressRange
    from OCP.RWGltf import RWGltf_CafWriter
    from OCP.RWMesh import RWMesh_CoordinateSystem
    from OCP.TColStd import TColStd_IndexedDataMapOfStringString
    from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    document = TDocStd_Document(TCollection_ExtendedString("cadgen-mesh"))
    XCAFDoc_DocumentTool.ShapeTool_s(document.Main()).AddShape(topods, False)
    with tempfile.TemporaryDirectory(prefix="cadgen-mesh-") as folder:
        path = Path(folder) / "component.glb"
        writer = RWGltf_CafWriter(TCollection_AsciiString(str(path)), True)
        writer.SetMergeFaces(False)
        converter = writer.ChangeCoordinateSystemConverter()
        converter.SetInputCoordinateSystem(RWMesh_CoordinateSystem.RWMesh_CoordinateSystem_Zup)
        converter.SetOutputCoordinateSystem(RWMesh_CoordinateSystem.RWMesh_CoordinateSystem_Zup)
        if not writer.Perform(document, TColStd_IndexedDataMapOfStringString(), Message_ProgressRange()):
            return None
        data = path.read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_length])
    binary = memoryview(data)[20 + json_length + 8:]

    def view(accessor_index: int, dtype, width: int) -> np.ndarray:
        accessor = gltf["accessors"][accessor_index]
        buffer_view = gltf["bufferViews"][accessor["bufferView"]]
        offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        return np.frombuffer(binary, dtype, accessor["count"] * width, offset)

    primitives = [primitive for mesh in gltf.get("meshes", []) for primitive in mesh["primitives"]]
    if len(primitives) != len(triangulated):
        return None
    # The writer places a located shape with a node transform; the component is
    # unlocated, so every node must be the identity for the arrays to be its own.
    for node in gltf.get("nodes", []):
        if any(name in node for name in ("matrix", "translation", "rotation", "scale")):
            matrix = node.get("matrix")
            if matrix is None or not np.allclose(matrix, np.eye(4).ravel()):
                return None
    faces = {}
    for (ordinal, _face, triangulation, _location), primitive in zip(triangulated, primitives):
        positions = view(primitive["attributes"]["POSITION"], np.float32, 3).reshape(-1, 3)
        normals = view(primitive["attributes"]["NORMAL"], np.float32, 3).reshape(-1, 3)
        index_accessor = gltf["accessors"][primitive["indices"]]
        dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}[index_accessor["componentType"]]
        triangles = view(primitive["indices"], dtype, 1).astype(np.uint32).reshape(-1, 3)
        if (len(positions) != triangulation.NbNodes() or len(triangles) != triangulation.NbTriangles()
                or ordinal in faces):
            return None
        faces[ordinal] = (positions, normals, triangles)
    return faces


def _faces_one_by_one(triangulated) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The same arrays read face by face: placed positions, normals from the exact surface
    turned to the face's orientation, and triangles wound to it."""
    from OCP.BRepLib import BRepLib_ToolTriangulatedShape
    from OCP.TopAbs import TopAbs_REVERSED

    faces = {}
    for ordinal, face, triangulation, location in triangulated:
        if ordinal in faces:
            continue
        if not triangulation.HasNormals():
            BRepLib_ToolTriangulatedShape.ComputeNormals_s(face, triangulation)
        transform = location.Transformation()
        count = triangulation.NbNodes()
        positions = np.empty((count, 3), np.float32)
        normals = np.empty((count, 3), np.float32)
        flip = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
        for node in range(1, count + 1):
            point = triangulation.Node(node).Transformed(transform)
            normal = triangulation.Normal(node).Transformed(transform)
            positions[node - 1] = (point.X(), point.Y(), point.Z())
            normals[node - 1] = (flip * normal.X(), flip * normal.Y(), flip * normal.Z())
        triangles = np.empty((triangulation.NbTriangles(), 3), np.uint32)
        for index in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(index).Get()
            triangles[index - 1] = (a - 1, c - 1, b - 1) if flip < 0 else (a - 1, b - 1, c - 1)
        faces[ordinal] = (positions, normals, triangles)
    return faces


def _edge_faces(topods, face_map, edge_map) -> list[list[int]]:
    """Each edge's adjacent face ordinals, deduplicated (a seam meets its face twice)."""
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer

    adjacent: list[list[int]] = [[] for _ in range(edge_map.Extent() + 1)]
    for face_ordinal in range(1, face_map.Extent() + 1):
        explorer = TopExp_Explorer(face_map.FindKey(face_ordinal), TopAbs_EDGE)
        while explorer.More():
            edge_ordinal = edge_map.FindIndex(explorer.Current())
            if edge_ordinal and face_ordinal not in adjacent[edge_ordinal]:
                adjacent[edge_ordinal].append(face_ordinal)
            explorer.Next()
    return adjacent


def _edge_polyline(edge, adjacent: list[int], face_map, faces, deflection: float, angle: float) -> np.ndarray | None:
    """The edge's points: its discretization on an adjacent face's mesh, else its own curve."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_TangentialDeflection
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    if BRep_Tool.Degenerated_s(edge):
        return None
    for face_ordinal in adjacent:
        if face_ordinal not in faces:
            continue
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(TopoDS.Face_s(face_map.FindKey(face_ordinal)), location)
        polygon = BRep_Tool.PolygonOnTriangulation_s(edge, triangulation, location) if triangulation is not None else None
        if polygon is None:
            continue
        nodes = polygon.Nodes()
        indices = np.fromiter((nodes.Value(k) - 1 for k in range(nodes.Lower(), nodes.Upper() + 1)),
                              np.int64, nodes.Length())
        return faces[face_ordinal][0][indices]
    # A free edge, or one no face meshed along: sample its own exact curve.
    curve = BRepAdaptor_Curve(edge)
    sampler = GCPnts_TangentialDeflection(curve, angle, deflection)
    if sampler.NbPoints() < 2:
        return None
    return np.array([[(point := sampler.Value(k)).X(), point.Y(), point.Z()]
                     for k in range(1, sampler.NbPoints() + 1)], np.float32)


def mesh_component(topods, surf_index: dict, *, surface_input: str, surface_object: str,
                   chord: float, angle: float) -> bytes:
    """Mesh one unlocated component and return its TESS v5 body.

    ``topods`` is a private decode of the component's BREP: meshing stores its
    triangulation on the shape, so it must not be a shape anything else holds.
    ``surf_index`` is the component's SURF index (its faces' colours and areas,
    its edges' classes), in the same ordinals.
    """
    from cadgen.store.meshes import encode_payload
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopoDS import TopoDS

    face_map, edge_map = _maps(topods)
    surf_faces, surf_edges = surf_index.get("faces", []), surf_index.get("edges", [])
    if len(surf_faces) != face_map.Extent() or len(surf_edges) != edge_map.Extent():
        raise MeshProductionError(
            f"the component's BREP has {face_map.Extent()} faces and {edge_map.Extent()} edges, "
            f"its SURF index {len(surf_faces)} and {len(surf_edges)}"
        )
    diagonal = _bounding_diagonal(topods)
    deflection = chord * diagonal
    BRepMesh_IncrementalMesh(topods, deflection, False, angle, True)
    triangulated = _triangulated_faces(topods, face_map)
    faces = (_faces_from_gltf(topods, triangulated) if triangulated else {})
    if faces is None:
        faces = _faces_one_by_one(triangulated)
    unmeshed = [row["ord"] for row in surf_faces
                if row["ord"] not in faces and float(row.get("area") or 0.0) > _NEGLIGIBLE_AREA * diagonal * diagonal]
    if unmeshed:
        listed = ", ".join(f"f{ordinal}" for ordinal in unmeshed[:8]) + (", ..." if len(unmeshed) > 8 else "")
        raise MeshProductionError(f"OCCT did not mesh {len(unmeshed)} face(s) of the component: {listed}")

    position_parts, normal_parts, face_ord_parts, index_parts, face_ranges = [], [], [], [], []
    vertex_base = index_start = 0
    for row in surf_faces:
        ordinal = row["ord"]
        positions, normals, triangles = faces.get(ordinal, (np.zeros((0, 3), np.float32),) * 2 + (np.zeros((0, 3), np.uint32),))
        position_parts.append(positions)
        normal_parts.append(normals)
        face_ord_parts.append(np.full(len(positions), ordinal, np.float32))
        index_parts.append(triangles.reshape(-1) + np.uint32(vertex_base))
        color = row.get("color")
        face_ranges.append({"ord": ordinal, "color": [float(c) for c in color] if color else None,
                            "indexStart": index_start, "indexCount": int(triangles.size)})
        vertex_base += len(positions)
        index_start += int(triangles.size)
    positions = np.concatenate(position_parts).astype("<f4") if position_parts else np.zeros((0, 3), "<f4")
    normals = np.concatenate(normal_parts).astype("<f4") if normal_parts else np.zeros((0, 3), "<f4")
    face_ords = np.concatenate(face_ord_parts).astype("<f4") if face_ord_parts else np.zeros(0, "<f4")
    indices = np.concatenate(index_parts).astype("<u4") if index_parts else np.zeros(0, "<u4")

    adjacent = _edge_faces(topods, face_map, edge_map)
    edges, edge_classes = [], []
    for row in surf_edges:
        ordinal, visibility = row["ord"], str(row.get("class") or "none")
        edge_classes.append([ordinal, visibility])
        polyline = _edge_polyline(TopoDS.Edge_s(edge_map.FindKey(ordinal)), adjacent[ordinal],
                                  face_map, faces, deflection, angle)
        if polyline is not None and len(polyline) >= 2:
            edges.append((ordinal, visibility, np.ascontiguousarray(polyline, "<f4").tobytes()))

    if len(positions):
        low, high = positions.min(axis=0), positions.max(axis=0)
    elif edges:
        points = np.concatenate([np.frombuffer(polyline, "<f4").reshape(-1, 3) for _, _, polyline in edges])
        low, high = points.min(axis=0), points.max(axis=0)
    else:
        low = high = np.zeros(3, np.float32)
    bounds = {"min": [float(v) for v in low], "max": [float(v) for v in high]}
    part_color = surf_index.get("partColor")
    return encode_payload(
        surface_input=surface_input, surface_object=surface_object, chord=chord, angle=angle,
        positions=positions.tobytes(), normals=normals.tobytes(), face_ords=face_ords.tobytes(),
        indices=indices.tobytes(), face_ranges=face_ranges, edges=edges, edge_classes=edge_classes,
        bounds=bounds, scale=float(diagonal),
        part_color=[float(c) for c in part_color] if part_color else None,
    )
