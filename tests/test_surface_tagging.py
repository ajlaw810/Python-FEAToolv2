"""
test_surface_tagging.py
=======================
Checkpoint 3.2: Topological Surface Tagging & Node Selection Mapping.

Verifies the Stage 3B extensions of CADGeometryPipeline (fea_geometry.py):

  1. Discovers all unique CAD surface tags for the block_with_hole fixture.
  2. Implements single-click CAD surface selection: given a picked boundary
     triangle index on the cylindrical bore, returns the parent surface tag
     and retrieves the full (corner + mid-edge) boundary node set.
  3. Geometrically verifies 100% of bore nodes satisfy the cylindrical
     radius equation (x-5)^2 + (y-1)^2 ≈ 0.5^2.
  4. Runs a C++ FEA solve using tag-based boundary condition application
     (fixed on the x=0 face, load on the x=10 face).
"""

import os

import numpy as np
import pytest

from cpp_python_project.core import FEASolver
from fea_geometry import CADGeometryPipeline

from test_step_mesh import STEP_FILE, generate_test_step_file

# Bore geometry (from fixture generation in test_step_mesh.py):
#   Box   : [0,10] x [0,2] x [0,2]
#   Bore  : cylinder radius 0.5, axis through (5, 1) along Z
BORE_CX = 5.0
BORE_CY = 1.0
BORE_R2 = 0.5**2


@pytest.fixture(scope="module")
def pipeline():
    """Generate the STEP fixture once and mesh it through the pipeline."""
    if not os.path.isfile(STEP_FILE):
        generate_test_step_file(STEP_FILE)

    with CADGeometryPipeline(STEP_FILE, mesh_size_min=0.5, mesh_size_max=1.0) as p:
        p.generate_mesh()
        yield p


# ---------------------------------------------------------------------------
# Checkpoint 3.2
# ---------------------------------------------------------------------------
def test_checkpoint_3_2_surface_tagging(pipeline):
    """Full topological surface tagging verification."""
    nodes = pipeline.get_nodes()
    elements = pipeline.get_elements()
    boundary_faces = pipeline.get_boundary_faces()
    boundary_tags = pipeline.get_boundary_surface_tags()

    n_nodes = len(nodes)
    n_faces = len(boundary_faces)

    print(f"\n=== Checkpoint 3.2 Diagnostics ===")

    # ------------------------------------------------------------------
    # Step 2: Discover surface tags and verify 7 unique
    # ------------------------------------------------------------------
    surface_tags = pipeline.get_surface_tags()
    n_surfaces = len(surface_tags)
    print(f"Surface tags         : {surface_tags}")
    assert n_surfaces == 7, (
        f"Expected 7 unique CAD surfaces, found {n_surfaces}"
    )
    assert surface_tags == sorted(surface_tags), "Surface tags not sorted"

    # Sanity: triangles dict integrity vs global boundary arrays
    total_tris = sum(len(v) for v in
                     [pipeline.get_triangles_for_surface_tag(t) for t in surface_tags])
    assert total_tris == n_faces, (
        f"Triangles dict total {total_tris} != boundary_faces {n_faces}"
    )

    # ------------------------------------------------------------------
    # Step 3: Identify the cylindrical bore tag geometrically
    # ------------------------------------------------------------------
    def tag_centroids(tag: int) -> np.ndarray:
        tris = pipeline.get_triangles_for_surface_tag(tag)  # (M, 3) 0-based
        return nodes[tris].mean(axis=1)  # (M, 3) centroid coords

    def tag_all_vertices_on_cylinder(tag: int) -> bool:
        """True if every triangle VERTEX of the surface lies on the bore."""
        tris = pipeline.get_triangles_for_surface_tag(tag)  # (M, 3) 0-based
        verts = nodes[tris]  # (M, 3, 3) -> flatten unique corners
        vx = verts[..., 0].ravel()
        vy = verts[..., 1].ravel()
        r2 = (vx - BORE_CX) ** 2 + (vy - BORE_CY) ** 2
        # NOTE: check vertices (on the cylinder), not centroids (which lie
        # strictly INSIDE the circle for a triangle inscribed on it).
        return bool(np.allclose(r2, BORE_R2, atol=1e-6))

    bore_tag = None
    for tag in surface_tags:
        if tag_all_vertices_on_cylinder(tag):
            bore_tag = tag
            break
    assert bore_tag is not None, "Could not identify cylindrical bore surface tag"
    print(f"Bore surface tag     : {bore_tag}")

    # Sanity: every other (planar) surface must NOT be on the cylinder.
    planar_tags = [t for t in surface_tags if t != bore_tag]
    for tag in planar_tags:
        assert not tag_all_vertices_on_cylinder(tag), (
            f"Planar surface {tag} incorrectly classified as bore"
        )

    # ------------------------------------------------------------------
    # Step 4: Single-click CAD surface selection logic
    # ------------------------------------------------------------------
    # Simulate a viewport pick: first boundary triangle belonging to the bore.
    bore_face_mask = boundary_tags == bore_tag
    picked_idx = int(np.argmax(bore_face_mask))

    recovered_tag = pipeline.get_surface_tag_for_triangle_index(picked_idx)
    assert recovered_tag == bore_tag, (
        f"Picked triangle {picked_idx} -> tag {recovered_tag}, "
        f"expected bore tag {bore_tag}"
    )

    # Retrieve the full boundary node set for the bore (corner + mid-edge).
    bore_nodes = pipeline.get_nodes_for_surface_tag(bore_tag)
    assert bore_nodes.ndim == 1, "bore_nodes should be 1D"
    assert bore_nodes.dtype == np.int32, f"bore_nodes dtype {bore_nodes.dtype}, expected int32"
    assert bore_nodes.min() >= 0, "bore_nodes must be 0-based"
    assert bore_nodes.max() < n_nodes, "bore_nodes out of range"
    assert np.all(bore_nodes[:-1] <= bore_nodes[1:]), "bore_nodes must be sorted"
    assert len(bore_nodes) > 0, "bore_nodes empty"

    # ------------------------------------------------------------------
    # Step 5: Geometric verification — 100% of bore nodes on the cylinder
    # ------------------------------------------------------------------
    bore_coords = nodes[bore_nodes]
    r2_all = (bore_coords[:, 0] - BORE_CX) ** 2 + (bore_coords[:, 1] - BORE_CY) ** 2
    max_dev = np.max(np.abs(r2_all - BORE_R2))
    on_cylinder = np.allclose(r2_all, BORE_R2, atol=1e-6)
    print(f"Bore nodes           : {len(bore_nodes)}")
    print(f"Max |r^2 - 0.25|     : {max_dev:.3e}")
    assert on_cylinder, (
        f"{np.sum(~np.isclose(r2_all, BORE_R2, atol=1e-6))} of "
        f"{len(bore_nodes)} bore nodes fail the cylinder equation"
    )

    # ------------------------------------------------------------------
    # Step 6: Tag-based BC faces — identify x=0 and x=10 faces
    # ------------------------------------------------------------------
    x0_tag = x10_tag = None
    for tag in surface_tags:
        centroids = tag_centroids(tag)
        if np.allclose(centroids[:, 0], 0.0, atol=1e-6):
            x0_tag = tag
        elif np.allclose(centroids[:, 0], 10.0, atol=1e-6):
            x10_tag = tag
    assert x0_tag is not None, "x=0 end-face tag not found"
    assert x10_tag is not None, "x=10 end-face tag not found"
    print(f"Fixed-face tag (x=0) : {x0_tag}")
    print(f"Loaded-face tag (x=10): {x10_tag}")

    # ------------------------------------------------------------------
    # Step 7: Tag-based BC application + C++ FEA solve
    # ------------------------------------------------------------------
    E = 210.0e9  # 210 GPa
    nu = 0.3

    fixed_node_ids = pipeline.get_nodes_for_surface_tag(x0_tag).astype(np.int32)
    loaded_node_ids = pipeline.get_nodes_for_surface_tag(x10_tag).astype(np.int32)
    assert len(fixed_node_ids) > 0, "No fixed nodes on x=0 face"
    assert len(loaded_node_ids) > 0, "No loaded nodes on x=10 face"

    total_force_z = -1000.0  # -1 kN total, downward along Z
    forces = np.zeros((len(loaded_node_ids), 3), dtype=np.float64)
    forces[:, 2] = total_force_z / len(loaded_node_ids)

    solver = FEASolver(nodes, elements, E, nu)
    solver.apply_fixed_bc(fixed_node_ids)
    solver.apply_point_loads(loaded_node_ids, forces)
    solver.solve()  # must complete without inverted-element exceptions

    displacements = solver.get_displacements()
    assert displacements.shape == (n_nodes, 3)
    max_abs_disp = np.max(np.abs(displacements))
    assert np.isfinite(max_abs_disp) and max_abs_disp > 0.0
    print(f"Max |displacement|   : {max_abs_disp:.6e} m")

    print("\n[CHECKPOINT 3.2 PASSED]: Topological Surface Tagging & CAD Selection Engine Verified!\n")


# ---------------------------------------------------------------------------
# Edge-case / API safety tests
# ---------------------------------------------------------------------------
def test_surface_tag_lookup_errors(pipeline):
    """Invalid triangle index and invalid surface tag must raise."""
    n_faces = len(pipeline.get_boundary_faces())

    # Out-of-range triangle indices
    with pytest.raises(IndexError):
        pipeline.get_surface_tag_for_triangle_index(-1)
    with pytest.raises(IndexError):
        pipeline.get_surface_tag_for_triangle_index(n_faces)

    # Invalid surface tag
    with pytest.raises(KeyError, match="not found"):
        pipeline.get_nodes_for_surface_tag(9999)
    with pytest.raises(KeyError, match="not found"):
        pipeline.get_triangles_for_surface_tag(9999)


def test_triangle_indices_are_0based_and_in_range(pipeline):
    """Every triangle node index must be a valid 0-based index into nodes."""
    nodes = pipeline.get_nodes()
    n_nodes = len(nodes)

    for tag in pipeline.get_surface_tags():
        tris = pipeline.get_triangles_for_surface_tag(tag)
        assert tris.ndim == 2 and tris.shape[1] == 3
        assert tris.dtype == np.int32
        assert tris.min() >= 0, f"triangles for tag {tag} not 0-based"
        assert tris.max() < n_nodes, f"triangles for tag {tag} out of range"

        # Do the triangle node IDs actually lie on the surface? Sample the
        # first triangle centroid and ensure it matches the surface tag
        # returned by the O(1) index lookup.
        first_face_global = tris[0]
        # Find the global boundary index of this triangle by matching rows.
        global_faces = pipeline.get_boundary_faces()
        match = np.where(
            (global_faces == first_face_global).all(axis=1)
        )[0]
        assert len(match) > 0, "triangle not found in global boundary_faces"
        global_idx = int(match[0])
        assert pipeline.get_surface_tag_for_triangle_index(global_idx) == tag