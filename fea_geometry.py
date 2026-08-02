"""
fea_geometry.py
===============
Stage 3A: CAD STEP file ingestion and high-order TET10 meshing using
Gmsh + OpenCASCADE.

This module provides the :class:`CADGeometryPipeline` class, which:

  1. Imports a ``.step`` / ``.stp`` CAD file via OpenCASCADE
     (``gmsh.model.occ.importShapes``).
  2. Generates a 2nd-order (quadratic) TET10 solid mesh with
     high-order surface geometry snapping enabled.
  3. Extracts:
       - 3D node coordinates  -> ``(N_nodes, 3)`` float64 NumPy array
       - TET10 connectivity   -> ``(N_elements, 10)`` int32 NumPy array
         (0-based indexing, C++ compatible)
       - Boundary metadata    -> 2D triangle faces + parent CAD surface
         tags (Stage 3B: face-to-surface mapping for UI selection)

The class is designed as a context manager so that ``gmsh.finalize()``
is always called safely, even if mesh generation raises an exception.

Dependencies: gmsh (Python API), numpy
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

import gmsh

# ---------------------------------------------------------------------------
# Module-level guard: Gmsh is a singleton; only one pipeline may be active
# at a time. This prevents context collisions across tests / UI sessions.
# ---------------------------------------------------------------------------
_active_pipeline: Optional["CADGeometryPipeline"] = None


class CADGeometryPipeline:
    """Import a STEP CAD file and generate a high-order TET10 mesh.

    Parameters
    ----------
    step_file : str
        Path to the ``.step`` / ``.stp`` CAD file to import.
    mesh_size_min : float, optional
        Minimum target mesh element size. If ``None`` (default), computed
        from the model bounding-box diagonal as ``max(diag * 0.03, 0.1)``.
    mesh_size_max : float, optional
        Maximum target mesh element size. If ``None`` (default), computed
        from the model bounding-box diagonal as ``max(diag * 0.08, 0.5)``.
    """

    # Gmsh element type code for a 10-node tetrahedron (TET10).
    _TET10_TYPE = 11

    def __init__(
        self,
        step_file: str,
        mesh_size_min: Optional[float] = None,
        mesh_size_max: Optional[float] = None,
    ) -> None:
        global _active_pipeline

        if _active_pipeline is not None:
            raise RuntimeError(
                "A CADGeometryPipeline is already active. Gmsh is a singleton; "
                "close the existing pipeline before creating a new one."
            )

        if not os.path.isfile(step_file):
            raise FileNotFoundError(f"STEP file not found: {step_file}")

        self.step_file = os.path.abspath(step_file)
        self.mesh_size_min = mesh_size_min
        self.mesh_size_max = mesh_size_max

        self._finalized = False
        self._mesh_generated = False

        # Extracted mesh data (populated by generate_mesh()).
        self._nodes: Optional[np.ndarray] = None
        self._elements: Optional[np.ndarray] = None
        self._boundary_faces: Optional[np.ndarray] = None
        self._boundary_surface_tags: Optional[np.ndarray] = None

        # Stage 3B: topological surface tagging maps (built by generate_mesh()).
        #   _boundary_face_all_nodes : (N_faces, 6) int32 0-based full quadratic
        #       node IDs per boundary triangle (corners + mid-edge nodes).
        #   _surface_tag_to_triangles : {tag: (M_i, 3) int32 0-based corner
        #       triangles} for PyVista face highlighting.
        #   _surface_tag_to_nodes : {tag: 1D int32 sorted unique node IDs
        #       (corner + mid-edge)} for tag-based BC application.
        #   _surface_tags : sorted list of unique CAD surface entity IDs.
        self._boundary_face_all_nodes: Optional[np.ndarray] = None
        self._surface_tag_to_triangles: Optional[dict] = None
        self._surface_tag_to_nodes: Optional[dict] = None
        self._surface_tags: Optional[list] = None

        # --- Initialize Gmsh and import the CAD model ----------------------
        gmsh.initialize()
        _active_pipeline = self

        try:
            gmsh.model.occ.importShapes(self.step_file)
            gmsh.model.occ.synchronize()

            # --- Bounding-box relative mesh sizing -------------------------
            # If the caller did not supply explicit mesh size bounds, compute
            # them from the model's bounding-box diagonal so the mesh density
            # scales with the part size. This keeps complex STEP models
            # (shafts, ball joints, etc.) in a responsive 2k-10k element
            # range instead of exploding to hundreds of thousands.
            if self.mesh_size_min is None or self.mesh_size_max is None:
                xmin, ymin, zmin, xmax, ymax, zmax = (
                    gmsh.model.getBoundingBox(-1, -1)
                )
                diag = float(np.sqrt(
                    (xmax - xmin) ** 2
                    + (ymax - ymin) ** 2
                    + (zmax - zmin) ** 2
                ))
                if self.mesh_size_min is None:
                    self.mesh_size_min = max(diag * 0.03, 0.1)
                if self.mesh_size_max is None:
                    self.mesh_size_max = max(diag * 0.08, 0.5)

            self.mesh_size_min = float(self.mesh_size_min)
            self.mesh_size_max = float(self.mesh_size_max)

            # --- High-order TET10 meshing configuration --------------------
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            # When auto-sizing (bounding-box relative) is used, the mesh is
            # intentionally coarse for interactive responsiveness. The
            # curvilinear high-order optimizer can abort the process on such
            # coarse meshes (inverted elements it cannot untangle), so we
            # disable it. Explicitly-supplied sizes (finer meshes) keep the
            # optimizer enabled.
            gmsh.option.setNumber(
                "Mesh.HighOrderOptimize",
                1 if (mesh_size_min is not None or mesh_size_max is not None) else 0,
            )
            gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size_max)

            # Prevent ultra-fine meshes from CAD-embedded characteristic
            # lengths and curvature. STEP files often carry very small
            # per-point/per-curve sizes that, when propagated by Gmsh's
            # defaults (``Mesh.MeshSizeFromPoints=1``,
            # ``Mesh.MeshSizeExtendFromBoundary=1``, and
            # ``Mesh.MeshSizeFromCurvature=1``), override the global min/max
            # bounds above and can generate millions of tiny TET10 elements
            # (slow, unresponsive GUI loading). We force the mesh to honor
            # our explicit ``mesh_size_min`` / ``mesh_size_max`` bounds.
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        except Exception:
            # Roll back gmsh initialization on import failure.
            self.close()
            raise

    # -----------------------------------------------------------------------
    # Context manager protocol: guarantees gmsh.finalize() on exit.
    # -----------------------------------------------------------------------
    def __enter__(self) -> "CADGeometryPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        # Return False so exceptions propagate normally.
        return False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def generate_mesh(self) -> None:
        """Generate the 3D solid mesh and extract all mesh arrays.

        If the high-order (curvilinear) optimization fails — which can happen
        on very coarse meshes of slender geometries where the initial linear
        mesh has inverted elements — we retry with ``Mesh.HighOrderOptimize=0``
        so the pipeline never crashes the host application.
        """
        if self._finalized:
            raise RuntimeError("Pipeline has been closed; cannot generate mesh.")

        try:
            gmsh.model.mesh.generate(3)
        except Exception:
            # High-order optimization failed (e.g. inverted elements on a
            # coarse mesh). Retry without the curvilinear optimizer so the
            # mesh is still produced and the app stays responsive.
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
            gmsh.model.mesh.generate(3)

        self._mesh_generated = True

        self._extract_nodes()
        self._extract_elements()
        self._extract_boundary_metadata()
        self._build_surface_maps()

    def get_nodes(self) -> np.ndarray:
        """Return node coordinates as ``(N_nodes, 3)`` float64 array."""
        self._require_mesh()
        assert self._nodes is not None
        return self._nodes

    def get_elements(self) -> np.ndarray:
        """Return TET10 connectivity as ``(N_elements, 10)`` int32 array.

        Uses 0-based node indexing for direct C++ compatibility.
        """
        self._require_mesh()
        assert self._elements is not None
        return self._elements

    def get_boundary_faces(self) -> np.ndarray:
        """Return 2D boundary triangle faces as ``(N_faces, 3)`` int32 array.

        Uses 0-based node indexing referencing the same node array returned
        by :meth:`get_nodes`. (Stage 3B: face-to-surface mapping.)
        """
        self._require_mesh()
        assert self._boundary_faces is not None
        return self._boundary_faces

    def get_boundary_surface_tags(self) -> np.ndarray:
        """Return parent CAD surface tag per boundary face as ``(N_faces,)`` int32.

        Each entry is the Gmsh elementary surface tag (dimension 2 entity)
        that owns the corresponding triangle in :meth:`get_boundary_faces`.
        (Stage 3B: enables clicking a face to select the entire CAD surface.)
        """
        self._require_mesh()
        assert self._boundary_surface_tags is not None
        return self._boundary_surface_tags

    # -----------------------------------------------------------------------
    # Stage 3B: Topological surface tagging & node selection API
    # -----------------------------------------------------------------------
    def get_surface_tags(self) -> list:
        """Return a sorted list of all unique CAD surface entity IDs."""
        self._require_mesh()
        assert self._surface_tags is not None
        return list(self._surface_tags)

    def get_nodes_for_surface_tag(self, tag: int) -> np.ndarray:
        """Return 1D int32 array of unique global 3D node IDs for a surface.

        Includes both corner and mid-edge (quadratic) nodes that touch the
        given CAD surface. Node IDs are 0-based, sorted ascending, and index
        directly into :meth:`get_nodes`.

        Raises ``KeyError`` if ``tag`` is not a valid surface entity ID.
        """
        self._require_mesh()
        self._require_surface_tag(tag)
        assert self._surface_tag_to_nodes is not None
        return self._surface_tag_to_nodes[tag]

    def get_triangles_for_surface_tag(self, tag: int) -> np.ndarray:
        """Return ``(M_i, 3)`` int32 boundary triangle node indices for a surface.

        Values are 0-based indices into the main :meth:`get_nodes` array,
        ready for PyVista face highlighting (e.g. ``pv.PolyData(nodes,
        faces=np.hstack([[3] + tri for tri in tris]))``).

        Raises ``KeyError`` if ``tag`` is not a valid surface entity ID.
        """
        self._require_mesh()
        self._require_surface_tag(tag)
        assert self._surface_tag_to_triangles is not None
        return self._surface_tag_to_triangles[tag]

    def get_surface_tag_for_triangle_index(self, triangle_idx: int) -> int:
        """Return the parent CAD surface tag for a picked boundary triangle.

        ``triangle_idx`` is a 0-based index into :meth:`get_boundary_faces`
        (e.g. from a viewport pick event). This is an O(1) lookup into the
        index-aligned ``_boundary_surface_tags`` array.

        Raises ``IndexError`` if ``triangle_idx`` is out of range.
        """
        self._require_mesh()
        assert self._boundary_surface_tags is not None
        n_faces = len(self._boundary_surface_tags)
        if triangle_idx < 0 or triangle_idx >= n_faces:
            raise IndexError(
                f"triangle_idx {triangle_idx} out of range [0, {n_faces - 1}]"
            )
        return int(self._boundary_surface_tags[triangle_idx])

    def close(self) -> None:
        """Safely finalize the Gmsh session.

        Idempotent: calling ``close()`` multiple times is a no-op after the
        first call. Safe to call even if mesh generation raised an exception.
        """
        global _active_pipeline

        if self._finalized:
            return

        try:
            gmsh.finalize()
        finally:
            self._finalized = True
            if _active_pipeline is self:
                _active_pipeline = None

    # -----------------------------------------------------------------------
    # Internal extraction helpers
    # -----------------------------------------------------------------------
    def _require_mesh(self) -> None:
        if self._finalized:
            raise RuntimeError("Pipeline has been closed; mesh data unavailable.")
        if not self._mesh_generated:
            raise RuntimeError(
                "Mesh has not been generated. Call generate_mesh() first."
            )

    def _extract_nodes(self) -> None:
        """Extract all 3D node coordinates as ``(N_nodes, 3)`` float64."""
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)
        coords_arr = np.asarray(coords, dtype=np.float64).reshape(n_nodes, 3)
        self._nodes = coords_arr

    def _extract_elements(self) -> None:
        """Extract TET10 (type 11) connectivity as ``(N_elements, 10)`` int32.

        Gmsh returns 1-based node IDs; we convert to 0-based for C++.
        """
        # gmsh.model.mesh.getElements(dim) returns 3 vectors:
        #   element_types, element_tags, node_tags
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(3)

        tet10_conn = []
        for etype, etags, ntags in zip(element_types, element_tags, node_tags):
            if etype != self._TET10_TYPE:
                continue
            n_elems = len(etags)
            conn = np.asarray(ntags, dtype=np.int32).reshape(n_elems, 10)
            tet10_conn.append(conn)

        if not tet10_conn:
            raise RuntimeError(
                "No TET10 (10-node tetrahedron) elements found in the mesh. "
                "Check that the CAD model is a 3D solid and meshing succeeded."
            )

        elements = np.vstack(tet10_conn)
        # Convert 1-based -> 0-based for C++ compatibility.
        elements -= 1
        self._elements = elements

    def _extract_boundary_metadata(self) -> None:
        """Extract 2D boundary triangles and their parent CAD surface tags.

        For each 2D elementary entity (surface), we fetch its mesh triangles
        and record the surface tag for every face. Node IDs are converted to
        0-based to match :meth:`get_nodes`.

        For quadratic (6-node) boundary triangles we retain BOTH:
          - the 3 corner nodes  -> ``_boundary_faces`` (PyVista rendering)
          - all 6 nodes         -> ``_boundary_face_all_nodes`` (Stage 3B:
            surface_tag_to_nodes must include corner + mid-edge nodes so that
            tag-based BC application constrains the full quadratic face).
        """
        faces_list: list[np.ndarray] = []
        all_nodes_list: list[np.ndarray] = []
        tags_list: list[np.ndarray] = []

        # Get all 2D entities (surfaces) in the model.
        surfaces = gmsh.model.getEntities(2)

        for dim, tag in surfaces:
            # Only surfaces that actually belong to the meshed model.
            try:
                # gmsh.model.mesh.getElements(2, tag) returns 3 vectors:
                #   element_types, element_tags, node_tags
                element_types, _, node_tags = gmsh.model.mesh.getElements(2, tag)
            except Exception:
                # Some surfaces may not be meshed (e.g. internal faces);
                # skip them.
                continue

            for etype, ntags in zip(element_types, node_tags):
                # Gmsh triangle type codes: 2 (3-node) and 9 (6-node).
                if etype == 2:
                    n_elems = len(ntags) // 3
                    faces = np.asarray(ntags, dtype=np.int32).reshape(n_elems, 3)
                    all_nodes = faces.copy()
                elif etype == 9:
                    n_elems = len(ntags) // 6
                    full = np.asarray(ntags, dtype=np.int32).reshape(n_elems, 6)
                    faces = full[:, :3]      # corner nodes only
                    all_nodes = full         # corners + mid-edge nodes
                else:
                    continue

                faces_list.append(faces)
                all_nodes_list.append(all_nodes)
                tags_list.append(np.full(faces.shape[0], tag, dtype=np.int32))

        if not faces_list:
            # No boundary faces found; store empty arrays.
            self._boundary_faces = np.empty((0, 3), dtype=np.int32)
            self._boundary_surface_tags = np.empty((0,), dtype=np.int32)
            self._boundary_face_all_nodes = np.empty((0, 6), dtype=np.int32)
            return

        boundary_faces = np.vstack(faces_list)
        boundary_all_nodes = np.vstack(all_nodes_list)
        boundary_tags = np.concatenate(tags_list)

        # Convert 1-based -> 0-based to match the primary node array.
        boundary_faces -= 1
        boundary_all_nodes -= 1

        self._boundary_faces = boundary_faces
        self._boundary_surface_tags = boundary_tags
        self._boundary_face_all_nodes = boundary_all_nodes

    def _build_surface_maps(self) -> None:
        """Build the Stage 3B bidirectional topological lookup maps.

        Populates:
          - ``_surface_tags`` : sorted list of unique CAD surface entity IDs.
          - ``_surface_tag_to_triangles`` : {tag: (M_i, 3) int32 0-based
            corner triangles} for PyVista face highlighting.
          - ``_surface_tag_to_nodes`` : {tag: 1D int32 sorted unique node IDs
            (corner + mid-edge)} for tag-based BC application.
        """
        assert self._boundary_surface_tags is not None
        assert self._boundary_faces is not None
        assert self._boundary_face_all_nodes is not None

        unique_tags = np.unique(self._boundary_surface_tags)
        self._surface_tags = [int(t) for t in unique_tags]

        tag_to_triangles: dict = {}
        tag_to_nodes: dict = {}

        for tag in unique_tags:
            tag_int = int(tag)
            mask = self._boundary_surface_tags == tag

            # (M_i, 3) corner triangles, already 0-based.
            tag_to_triangles[tag_int] = self._boundary_faces[mask]

            # All 6 quadratic nodes per face -> unique sorted 0-based IDs.
            all_nodes = self._boundary_face_all_nodes[mask]
            tag_to_nodes[tag_int] = np.unique(all_nodes).astype(np.int32)

        self._surface_tag_to_triangles = tag_to_triangles
        self._surface_tag_to_nodes = tag_to_nodes

    def _require_surface_tag(self, tag: int) -> None:
        """Raise KeyError if ``tag`` is not a valid CAD surface entity ID."""
        assert self._surface_tags is not None
        if tag not in self._surface_tags:
            raise KeyError(
                f"Surface tag {tag} not found. Available tags: {self._surface_tags}"
            )
