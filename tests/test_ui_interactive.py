"""
test_ui_interactive.py
======================
Checkpoint 4.1: PyQt6 + PyVista 3D Viewport & Interactive CAD Surface Picking.

Verifies the Stage 4A UI canvas (app_ui.py):

  1. The application window (FEAAppMainWindow) instantiates cleanly with
     an embedded pyvistaqt.QtInteractor in a headless/offscreen test
     environment.
  2. Loads tests/fixtures/block_with_hole.step through
     CADGeometryPipeline, extracts the TET10 mesh, and renders the outer
     CAD surface as an interactive PyVista mesh actor.
  3. Programmatically triggers/simulates a cell-pick event by calling the
     public selection API with a deterministic boundary triangle index,
     and asserts:
       - The picked face index resolves to its parent CAD surface tag.
       - The highlighted overlay actor receives 100% of the surface
         triangles matching that CAD tag.
  4. Prints the Checkpoint 4.1 PASSED message.
"""

import os

import numpy as np
import pytest

# --- Headless / off-screen Qt + VTK configuration --------------------------
# Must be set before importing pyvista / QtInteractor (app_ui does this, but
# set explicitly here to be defensive for any local imports).
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app_ui  # noqa: E402
from test_step_mesh import STEP_FILE, generate_test_step_file  # noqa: E402


@pytest.fixture(scope="module")
def app_window():
    """Instantiate the FEA app window and load the fixture STEP file."""
    if not os.path.isfile(STEP_FILE):
        generate_test_step_file(STEP_FILE)

    # Reuse an existing QApplication if pytest has already created one.
    from PyQt6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])

    window = app_ui.FEAAppMainWindow()
    window.show()

    mesh_data = window.load_step_file(STEP_FILE)
    assert mesh_data is not None, "load_step_file returned None"

    yield window

    window.close()


# ---------------------------------------------------------------------------
# Test 1: Window instantiates cleanly + renders the CAD surface
# ---------------------------------------------------------------------------
def test_window_instantiates_and_renders(app_window):
    """The main window builds a QtInteractor and renders the surface mesh."""
    window = app_window

    md = window.get_mesh_data()
    assert md is not None
    assert md.n_nodes > 0
    assert md.n_elements > 0
    assert md.n_boundary_faces > 0
    assert len(md.surface_tags) == 7, (
        f"Expected 7 CAD surfaces, got {len(md.surface_tags)}"
    )

    # The viewport must own a QtInteractor.
    assert window.plotter is not None

    # The boundary-surface Polydata must have been built from the mesh. This
    # is available in both interactive (GL actor) and headless (data-only)
    # modes — Checkpoint 4.1 verifies selection logic, which operates on data.
    surface_mesh = window.get_surface_mesh()
    assert surface_mesh is not None, "CAD surface polydata was not created"
    assert surface_mesh.n_cells == md.n_boundary_faces, (
        f"Surface mesh {surface_mesh.n_cells} cells != "
        f"boundary_faces {md.n_boundary_faces}"
    )

    print(f"\n=== Checkpoint 4.1 Diagnostics (UI Canvas) ===")
    print(f"Interactive picking  : {window.is_picking_enabled()}")
    print(f"Nodes                : {md.n_nodes}")
    print(f"TET10 elements       : {md.n_elements}")
    print(f"Boundary faces       : {md.n_boundary_faces}")
    print(f"Surface tags         : {md.surface_tags}")
    print(f"QtInteractor active  : {window.plotter is not None}")


# ---------------------------------------------------------------------------
# Checkpoint 4.1: Single-Click CAD Surface Selection
# ---------------------------------------------------------------------------
def test_single_click_cad_surface_picking(app_window):
    """Simulate a cell pick and verify parent-surface resolution + 100% overlay."""
    window = app_window
    md = window.get_mesh_data()
    assert md is not None

    # Pick a deterministic boundary triangle. Use the FIRST surface tag's
    # first boundary face so we exercise a real CAD face (not just tag 0).
    tag = md.surface_tags[0]
    faces_mask = md.boundary_surface_tags == tag
    picked_idx = int(np.argmax(faces_mask))

    print(f"Picked face index    : {picked_idx}")
    print(f"Expected surface tag : {tag}")

    # --- Trigger the "pick" through the same callback the viewport uses. ---
    # This exercises the full cell-pick path without requiring a mouse event.
    window._handle_cell_pick_callback(_make_picked_mesh(md, picked_idx))

    # Assertion A: the picked face resolves to its parent CAD surface tag.
    selected_tag = window.get_selected_surface_tag()
    assert selected_tag == tag, (
        f"Picked face {picked_idx} resolved to tag {selected_tag}, "
        f"expected {tag}"
    )
    print(f"Resolved tag         : {selected_tag}")

    # Assertion B: the highlight overlay holds 100% of that tag's triangles.
    overlay_tris = window.get_highlight_overlay_triangles()
    assert overlay_tris is not None
    expected_tris = md.surface_tag_to_triangles[tag]

    assert overlay_tris.shape == expected_tris.shape, (
        f"Overlay {overlay_tris.shape} != expected {expected_tris.shape}"
    )
    assert np.array_equal(overlay_tris, expected_tris), (
        "Overlay triangles do not EXACTLY match the full CAD surface "
        "triangle set (100% coverage required)."
    )
    print(f"Overlay triangles    : {len(overlay_tris)} "
          f"(100% of tag {tag})")

    # Assertion C: sidebar stats match.
    stats = window.get_selected_surface_stats()
    assert stats is not None
    assert stats["tag"] == tag
    assert stats["n_triangles"] == len(expected_tris)
    print(f"Surface node count   : {stats['n_nodes']}")

    # Assertion D: the highlight overlay Polydata was constructed and holds
    # the same 100% triangle set (available in interactive + headless modes).
    highlight_overlay = window.get_highlight_overlay()
    assert highlight_overlay is not None, "Highlight overlay was not created"
    assert highlight_overlay.n_cells == len(expected_tris), (
        f"Highlight overlay {highlight_overlay.n_cells} cells != "
        f"expected {len(expected_tris)}"
    )

    # When a real GL viewport is available, the overlay is also registered as
    # a named actor in the renderer.
    if window.is_picking_enabled():
        assert app_ui.FEAAppMainWindow.HIGHLIGHT_NAME in \
            window.plotter.renderer.actors, (
                "Highlight overlay actor not present in renderer."
            )

    print("\n[CHECKPOINT 4.1 PASSED]: PyQt6 + PyVista 3D Viewport & "
          "Interactive CAD Surface Picking Verified!")


# ---------------------------------------------------------------------------
# Test 2: Window instantiates with NO file loaded (clean startup)
# ---------------------------------------------------------------------------
def test_window_clean_startup_no_file():
    """App must instantiate without error before any file is loaded."""
    from PyQt6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])

    window = app_ui.FEAAppMainWindow()
    try:
        assert window.get_mesh_data() is None
        assert window.get_selected_surface_tag() is None
        assert window.get_highlight_overlay_triangles() is None
    finally:
        window.close()


# ---------------------------------------------------------------------------
# Helper: build a minimal "picked mesh" mimicking the callback object
# ---------------------------------------------------------------------------
def _make_picked_mesh(md: app_ui.MeshData, picked_face_idx: int):
    """Build a fake picked-mesh carrying vtkOriginalCellIds.

    The real QtInteractor callback passes a picked PolyData whose cell-data
    'vtkOriginalCellIds' maps the picked cell back to the original boundary
    face index. We synthesize the exact same structure so the callback's
    face-index recovery is exercised for real.
    """
    import pyvista as pv

    # A single triangle picked: reuse the picked boundary face's 3 corner
    # node indices so the geometry is consistent with the base mesh.
    tri = md.boundary_faces[picked_face_idx]

    faces = np.array([3, tri[0], tri[1], tri[2]], dtype=np.int64)
    picked_mesh = pv.PolyData(md.nodes, faces=faces)
    picked_mesh.cell_data["vtkOriginalCellIds"] = np.array(
        [picked_face_idx], dtype=np.int64
    )
    return picked_mesh