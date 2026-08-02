"""
test_step_mesh.py
=================
Checkpoint 3.1: STEP Import & High-Order TET10 Meshing.

Verifies the CAD STEP ingestion pipeline (fea_geometry.py):

  1. Programmatically generates a test STEP file (rectangular block with
     a cylindrical through-hole) via Gmsh OpenCASCADE.
  2. Processes the STEP through CADGeometryPipeline to generate a
     2nd-order TET10 mesh.
  3. Passes the extracted nodes/elements into the C++ FEASolver, applies
     boundary conditions, and calls .solve().
  4. Asserts mesh array shapes, dtype, 0-based indexing, boundary metadata
     (Stage 3B), and that the C++ solver completes without non-positive
     volume / inverted element exceptions.
"""

import os

import numpy as np
import pytest

import gmsh

from cpp_python_project.core import FEASolver
from fea_geometry import CADGeometryPipeline

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
STEP_FILE = os.path.join(FIXTURES_DIR, "block_with_hole.step")


def generate_test_step_file(output_path: str) -> str:
    """Programmatically create a STEP fixture: block with a cylindrical hole.

    Geometry:
      - Box   : 10 x 2 x 2, spanning [0,10] x [0,2] x [0,2]
      - Hole  : cylinder radius 0.5, axis along X through (5, 1, 1),
                length 12 (extends beyond block in -Z and +Z)
      - Result: block minus cylinder -> a through-hole bore.

    Returns the path to the written .step file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.occ.addBox(0, 0, 0, 10, 2, 2, tag=1)
        gmsh.model.occ.addCylinder(5, 1, -1, 0, 0, 4, 0.5, tag=2)
        gmsh.model.occ.cut(
            [(3, 1)], [(3, 2)],
            removeObject=True,
            removeTool=True,
        )
        gmsh.model.occ.synchronize()
        gmsh.write(output_path)
    finally:
        gmsh.finalize()

    assert os.path.isfile(output_path), f"Failed to write STEP file: {output_path}"
    return output_path


@pytest.fixture(scope="module")
def step_file() -> str:
    """Module-scoped fixture: generate the STEP file once per test session."""
    if not os.path.isfile(STEP_FILE):
        generate_test_step_file(STEP_FILE)
    return STEP_FILE


# ---------------------------------------------------------------------------
# Checkpoint 3.1
# ---------------------------------------------------------------------------
def test_checkpoint_3_1_step_tet10_mesh(step_file: str):
    """Full pipeline: STEP import -> TET10 mesh -> C++ FEASolver solve."""
    with CADGeometryPipeline(
        step_file,
        mesh_size_min=0.5,
        mesh_size_max=1.0,
    ) as pipeline:
        pipeline.generate_mesh()

        nodes = pipeline.get_nodes()
        elements = pipeline.get_elements()
        boundary_faces = pipeline.get_boundary_faces()
        boundary_tags = pipeline.get_boundary_surface_tags()

    n_nodes = len(nodes)
    n_elements = len(elements)
    n_faces = len(boundary_faces)

    print(f"\n=== Checkpoint 3.1 Diagnostics ===")
    print(f"STEP file            : {step_file}")
    print(f"Nodes                : {n_nodes}")
    print(f"TET10 elements       : {n_elements}")
    print(f"Boundary faces       : {n_faces}")
    print(f"Unique surface tags  : {len(np.unique(boundary_tags))}")

    # ------------------------------------------------------------------
    # Assertion 1: nodes array
    # ------------------------------------------------------------------
    assert nodes.ndim == 2, f"nodes.ndim == {nodes.ndim}, expected 2"
    assert nodes.shape[1] == 3, f"nodes.shape = {nodes.shape}, expected (N, 3)"
    assert nodes.dtype == np.float64, f"nodes dtype {nodes.dtype}, expected float64"

    # ------------------------------------------------------------------
    # Assertion 2: elements array (TET10)
    # ------------------------------------------------------------------
    assert elements.ndim == 2, f"elements.ndim == {elements.ndim}, expected 2"
    assert elements.shape[1] == 10, (
        f"elements.shape = {elements.shape}, expected (N, 10)"
    )
    assert elements.dtype == np.int32, (
        f"elements dtype {elements.dtype}, expected int32"
    )

    # Verify 0-based indexing for C++ compatibility
    assert elements.min() >= 0, (
        f"elements.min() = {elements.min()}, expected >= 0 (0-based)"
    )
    assert elements.max() < n_nodes, (
        f"elements.max() = {elements.max()} >= n_nodes = {n_nodes}"
    )

    # Verify each element has 10 unique node IDs (valid TET10, no duplicate
    # corner/mid nodes that would indicate corrupted connectivity).
    for e in range(n_elements):
        uniq = np.unique(elements[e])
        assert len(uniq) == 10, (
            f"Element {e} has {len(uniq)} unique nodes, expected 10"
        )

    # ------------------------------------------------------------------
    # Assertion 3: boundary metadata (Stage 3B compatibility)
    # ------------------------------------------------------------------
    assert boundary_faces.ndim == 2, (
        f"boundary_faces.ndim == {boundary_faces.ndim}, expected 2"
    )
    assert boundary_faces.shape[1] == 3, (
        f"boundary_faces.shape = {boundary_faces.shape}, expected (N, 3)"
    )
    assert boundary_faces.dtype == np.int32, (
        f"boundary_faces dtype {boundary_faces.dtype}, expected int32"
    )

    # 0-based indexing referencing the same node array as get_nodes()
    assert boundary_faces.min() >= 0, (
        f"boundary_faces.min() = {boundary_faces.min()}, expected >= 0 (0-based)"
    )
    assert boundary_faces.max() < n_nodes, (
        f"boundary_faces.max() = {boundary_faces.max()} >= n_nodes = {n_nodes}"
    )
    assert len(boundary_tags) == n_faces, (
        f"boundary_tags length {len(boundary_tags)} != boundary_faces {n_faces}"
    )
    assert boundary_tags.dtype == np.int32

    # A block with a cylindrical through-hole has 7 boundary surfaces:
    #   6 planar faces (box) + 1 cylindrical bore.
    n_unique_surfaces = len(np.unique(boundary_tags))
    assert n_unique_surfaces == 7, (
        f"Expected 7 unique CAD surfaces (6 box faces + 1 bore), "
        f"found {n_unique_surfaces}"
    )

    # ------------------------------------------------------------------
    # Assertion 4: C++ FEASolver on the STEP mesh
    # ------------------------------------------------------------------
    E = 210.0e9  # 210 GPa
    nu = 0.3

    solver = FEASolver(nodes, elements, E, nu)

    # Fixed BC: all nodes on the face x = 0
    fixed_node_ids = np.where(np.abs(nodes[:, 0]) < 1e-6)[0].astype(np.int32)
    assert len(fixed_node_ids) > 0, "No fixed nodes found on x = 0 face"

    # Load BC: downward force on the face x = 10
    loaded_node_ids = np.where(np.abs(nodes[:, 0] - 10.0) < 1e-6)[0].astype(
        np.int32
    )
    assert len(loaded_node_ids) > 0, "No loaded nodes found on x = 10 face"

    total_force_y = -1000.0  # -1 kN total, distributed over loaded nodes
    forces = np.zeros((len(loaded_node_ids), 3), dtype=np.float64)
    forces[:, 1] = total_force_y / len(loaded_node_ids)

    solver.apply_fixed_bc(fixed_node_ids)
    solver.apply_point_loads(loaded_node_ids, forces)

    # Must complete without throwing non-positive volume / inverted element
    # exceptions (Tet10Element constructor validates each element volume > 0).
    solver.solve()

    displacements = solver.get_displacements()
    assert displacements.shape == (n_nodes, 3), (
        f"displacements.shape = {displacements.shape}, expected ({n_nodes}, 3)"
    )

    # Sanity: max deflection should be finite and reasonable
    max_abs_disp = np.max(np.abs(displacements))
    assert np.isfinite(max_abs_disp), "Displacements contain NaN/Inf"
    assert max_abs_disp > 0.0, "Zero displacement — check boundary conditions"
    print(f"Max |displacement|   : {max_abs_disp:.6e} m")

    print("\n[CHECKPOINT 3.1 PASSED]: STEP Import & High-Order TET10 Meshing Verified!\n")


# ---------------------------------------------------------------------------
# Pipeline lifecycle / safety tests
# ---------------------------------------------------------------------------
def test_pipeline_close_is_idempotent(step_file: str):
    """close() may be called multiple times without crashing."""
    pipeline = CADGeometryPipeline(
        step_file, mesh_size_min=0.5, mesh_size_max=1.0
    )
    try:
        pipeline.generate_mesh()

        pipeline.close()
        pipeline.close()  # second call must be a no-op
        assert pipeline._finalized is True
    finally:
        # Guarantee the singleton guard is released even on failure.
        pipeline.close()


def test_active_pipeline_guard(step_file: str):
    """Creating a second pipeline while one is active raises RuntimeError."""
    pipeline = CADGeometryPipeline(
        step_file, mesh_size_min=0.5, mesh_size_max=1.0
    )
    try:
        with pytest.raises(RuntimeError, match="already active"):
            CADGeometryPipeline(
                step_file, mesh_size_min=0.5, mesh_size_max=1.0
            )
    finally:
        pipeline.close()


def test_getters_before_mesh_raise(step_file: str):
    """Accessing mesh data before generate_mesh() raises RuntimeError."""
    with CADGeometryPipeline(
        step_file, mesh_size_min=0.5, mesh_size_max=1.0
    ) as pipeline:
        with pytest.raises(RuntimeError, match="generate_mesh"):
            _ = pipeline.get_nodes()
        with pytest.raises(RuntimeError, match="generate_mesh"):
            _ = pipeline.get_elements()
        with pytest.raises(RuntimeError, match="generate_mesh"):
            _ = pipeline.get_boundary_faces()
        with pytest.raises(RuntimeError, match="generate_mesh"):
            _ = pipeline.get_boundary_surface_tags()