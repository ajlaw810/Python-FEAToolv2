"""
app_ui.py
=========
Stage 4A: PyQt6 + PyVista 3D Viewport & Interactive CAD Surface Selection Canvas.

This module implements the main desktop application window for the FEA
toolkit:

  1. **Main Window Architecture** (PyQt6 + PyVista)
     - ``PyQt6.QMainWindow`` with a split view:
         * Left  : control sidebar (File Loading, CAD Info, Surface
                   Selection Details, Solver Controls).
         * Right : embedded 3D viewport via ``pyvistaqt.QtInteractor``.

  2. **STEP File Loading & Viewport Rendering**
     - "Load STEP File" button opens a ``.step`` / ``.stp`` file dialog.
     - Runs the file through :class:`CADGeometryPipeline` to generate the
       high-order TET10 mesh, extracts the mesh arrays **immediately**
       (releasing the gmsh singleton), and renders the outer CAD surface
       as an interactive PyVista mesh actor.

  3. **Single-Click CAD Surface Selection (Approach A)**
     - ``enable_cell_picking`` on the ``QtInteractor``.
     - On pick: recover the picked triangle index → query
       ``get_surface_tag_for_triangle_index`` → highlight ALL triangles of
       that CAD surface tag as a translucent red overlay → display tag ID,
       triangle count, node count in the sidebar.

Gmsh is a singleton; the pipeline must be closed immediately after mesh
extraction so the app never holds the gmsh session while idle. All
selection logic operates on extracted NumPy arrays / dicts only.

Headless / no-GPU environments
------------------------------
When no OpenGL pixel format / interacting render window is available
(e.g. ``pytest`` under ``QT_QPA_PLATFORM=offscreen``, CI without a GPU),
``enable_cell_picking`` raises ``RuntimeError("This plotting window is not
interactive.")`` and any GL call (``add_mesh``, ``reset_camera``, ``render``)
may abort ``vtkWin32OpenGLRenderWindow`` with an access violation.

We therefore use "picking enabled" as the reliable signal that a real GL
viewport exists. In that case we render + wire up the interactive picker.
Otherwise we degrade gracefully: all *selection logic* still runs on the
extracted NumPy data (which is what Checkpoint 4.1 verifies) while GL
rendering is skipped. Interactive desktop use is unaffected.

Offscreen mode is NEVER configured by this module. When run directly as a
desktop app (``python app_ui.py``), this module strips any offscreen
environment variables that may have leaked from the environment and forces
PyVista desktop interactive mode. When imported as a module (e.g. by headless
pytest tests), the importing process's environment is left untouched so tests
can configure ``QT_QPA_PLATFORM=offscreen`` themselves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Desktop interactive mode is the default. When run directly as a desktop
# app (python app_ui.py), strip any offscreen env vars that may have leaked
# from the environment. When imported as a module (e.g. by headless pytest
# tests), leave the importing process's environment untouched.
if __name__ == "__main__":
    os.environ.pop("QT_QPA_PLATFORM", None)
    os.environ.pop("PYVISTA_OFF_SCREEN", None)
os.environ["VTK_SILENT_WARNINGS"] = "1"

# ---------------------------------------------------------------------------
# VTK render-window configuration (MUST be set before importing pyvista /
# pyvistaqt).
#
# On Windows, pyvistaqt.QtInteractor embeds a vtkWin32OpenGLRenderWindow,
# which can fail to bind to the native Win32 device context:
#   "vtkWin32OpenGLRenderWindow: failed to get valid pixel format"
#   "Unable to find a valid OpenGL 3.2 or later implementation."
#
# Disabling global multisampling (MSAA) is the standard workaround: it stops
# VTK from requesting a multisampled pixel format that the Win32 DC cannot
# provide, letting the render window negotiate a valid single-sample format.
# ---------------------------------------------------------------------------
import vtk  # noqa: E402
# The static MSAA cap is defined on vtkOpenGLRenderWindow (the base of the
# Win32 render window); vtkRenderWindow itself does not expose it.
vtk.vtkOpenGLRenderWindow.SetGlobalMaximumNumberOfMultiSamples(0)

# ---------------------------------------------------------------------------
# OpenGL surface format (MUST be set before importing pyvista / pyvistaqt).
#
# We also force an explicit OpenGL 3.2+ core-profile surface with a 24-bit
# depth and 8-bit stencil buffer so Qt creates a context VTK can adopt.
# ---------------------------------------------------------------------------
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QSurfaceFormat  # noqa: E402

_fmt = QSurfaceFormat()
_fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
_fmt.setVersion(3, 2)
_fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
_fmt.setDepthBufferSize(24)
_fmt.setStencilBufferSize(8)
QSurfaceFormat.setDefaultFormat(_fmt)

import pyvista as pv  # noqa: E402

# Force desktop interactive rendering when run as a desktop app. When
# imported as a module (headless pytest), leave PyVista's default alone so
# the test's offscreen configuration is preserved.
if __name__ == "__main__":
    pv.OFF_SCREEN = False

from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pyvistaqt import QtInteractor  # noqa: E402

from fea_geometry import CADGeometryPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# MeshData: immutable snapshot of the generated TET10 mesh + surface maps.
# ---------------------------------------------------------------------------
@dataclass
class MeshData:
    """Extracted mesh data after ``CADGeometryPipeline.generate_mesh()``.

    The pipeline (and therefore the gmsh singleton) is closed immediately
    after extraction; this snapshot holds every array needed for rendering
    and surface picking.
    """

    nodes: np.ndarray
    elements: np.ndarray
    boundary_faces: np.ndarray
    boundary_surface_tags: np.ndarray
    surface_tags: List[int]
    surface_tag_to_triangles: Dict[int, np.ndarray] = field(default_factory=dict)
    surface_tag_to_nodes: Dict[int, np.ndarray] = field(default_factory=dict)
    step_file: str = ""

    # -- Convenience -------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_boundary_faces(self) -> int:
        return len(self.boundary_faces)


def _build_polydata(mesh_data: MeshData, faces: Optional[np.ndarray] = None):
    """Build a ``pv.PolyData`` from node coords + (M,3) triangle array.

    ``faces`` defaults to the full boundary face set. The resulting
    PolyData has one cell per triangle, index-aligned with the input.
    """
    if faces is None:
        faces = mesh_data.boundary_faces
    n_cells = len(faces)
    # VTK face encoding: 3, n0, n1, n2  per triangle.
    faces_flat = np.empty(n_cells * 4, dtype=np.int64)
    faces_flat[0::4] = 3
    faces_flat[1::4] = faces[:, 0]
    faces_flat[2::4] = faces[:, 1]
    faces_flat[3::4] = faces[:, 2]
    return pv.PolyData(mesh_data.nodes, faces=faces_flat)


def build_mesh_data(step_file: str,
                    mesh_size_min: Optional[float] = None,
                    mesh_size_max: Optional[float] = None) -> MeshData:
    """Run a STEP file through CADGeometryPipeline and return the snapshot.

    The pipeline is created and closed inside this function so the gmsh
    singleton is never held by the UI. Raises ``FileNotFoundError`` if the
    file is missing and propagates meshing failures from the pipeline.

    When ``mesh_size_min`` / ``mesh_size_max`` are ``None`` (default), the
    pipeline computes bounding-box relative sizes so the mesh density scales
    with the part size — keeping complex STEP models in a responsive
    2k-10k element range.
    """
    if not os.path.isfile(step_file):
        raise FileNotFoundError(f"STEP file not found: {step_file}")

    with CADGeometryPipeline(
        step_file,
        mesh_size_min=mesh_size_min,
        mesh_size_max=mesh_size_max,
    ) as pipeline:
        pipeline.generate_mesh()

        nodes = pipeline.get_nodes()
        elements = pipeline.get_elements()
        boundary_faces = pipeline.get_boundary_faces()
        boundary_tags = pipeline.get_boundary_surface_tags()
        surface_tags = pipeline.get_surface_tags()

        tag_to_tris: Dict[int, np.ndarray] = {}
        tag_to_nodes: Dict[int, np.ndarray] = {}
        for tag in surface_tags:
            tag_to_tris[tag] = pipeline.get_triangles_for_surface_tag(tag)
            tag_to_nodes[tag] = pipeline.get_nodes_for_surface_tag(tag)

    return MeshData(
        nodes=nodes,
        elements=elements,
        boundary_faces=boundary_faces,
        boundary_surface_tags=boundary_tags,
        surface_tags=surface_tags,
        surface_tag_to_triangles=tag_to_tris,
        surface_tag_to_nodes=tag_to_nodes,
        step_file=os.path.abspath(step_file),
    )


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------
class FEAAppMainWindow(QMainWindow):
    """Main desktop application: split sidebar + PyVista 3D viewport."""

    HIGHLIGHT_NAME = "surface_highlight"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FEA Toolkit — Stage 4A: 3D CAD Surface Selection")
        self.resize(1280, 800)

        # --- Application state --------------------------------------------
        self._mesh_data: Optional[MeshData] = None
        self._selected_surface_tag: Optional[int] = None
        self._highlight_overlay_triangles: Optional[np.ndarray] = None
        self._surface_mesh_actor = None
        self._surface_polydata: Optional[pv.PolyData] = None
        self._highlight_overlay: Optional[pv.PolyData] = None
        self._picking_enabled = False

        # --- Central layout -----------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_viewport())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 960])
        self.setCentralWidget(splitter)

    # ----------------------------------------------------------------------
    # UI construction
    # ----------------------------------------------------------------------
    def _build_control_panel(self) -> QWidget:
        """Left sidebar with the four control group boxes."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. File Loading ---------------------------------------------------
        file_box = QGroupBox("File Loading")
        file_layout = QVBoxLayout(file_box)
        self.load_button = QPushButton("Load STEP File")
        self.load_button.clicked.connect(self._on_load_clicked)
        file_layout.addWidget(self.load_button)
        self.file_path_label = QLabel("No file loaded")
        self.file_path_label.setWordWrap(True)
        file_layout.addWidget(self.file_path_label)
        layout.addWidget(file_box)

        # 2. CAD Info -------------------------------------------------------
        cad_box = QGroupBox("CAD Info")
        cad_layout = QVBoxLayout(cad_box)
        self.nodes_label = QLabel("Nodes      : —")
        self.elements_label = QLabel("TET10 mesh : —")
        self.faces_label = QLabel("Surf faces : —")
        self.tags_label = QLabel("Surf tags  : —")
        for lbl in (self.nodes_label, self.elements_label,
                    self.faces_label, self.tags_label):
            cad_layout.addWidget(lbl)
        layout.addWidget(cad_box)

        # 3. Surface Selection Details --------------------------------------
        sel_box = QGroupBox("Surface Selection Details")
        sel_layout = QVBoxLayout(sel_box)
        self.sel_tag_label = QLabel("Selected CAD surface : —")
        self.sel_tri_label = QLabel("Triangle count       : —")
        self.sel_node_label = QLabel("Node count           : —")
        self.sel_hint_label = QLabel("Tip: left-click a surface triangle in "
                                     "the 3D view to select its CAD surface.")
        self.sel_hint_label.setWordWrap(True)
        for lbl in (self.sel_tag_label, self.sel_tri_label,
                    self.sel_node_label, self.sel_hint_label):
            sel_layout.addWidget(lbl)
        layout.addWidget(sel_box)

        # 4. Solver Controls (placeholder for Stage 5) ----------------------
        solver_box = QGroupBox("Solver Controls")
        solver_layout = QVBoxLayout(solver_box)
        run_btn = QPushButton("Run FEA Solve")
        run_btn.setEnabled(False)
        run_btn.setToolTip("Stage 5: solver integration pending.")
        solver_layout.addWidget(run_btn)
        solver_layout.addWidget(QLabel("Stage 5: solver integration pending."))
        layout.addWidget(solver_box)

        layout.addStretch(1)

        # Wrap in a scroll area for small resolutions.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        return scroll

    def _build_viewport(self) -> QWidget:
        """Right pane: embedded PyVista 3D viewport.

        Attempts to enable interactive cell picking. In headless / no-GPU
        contexts this raises, which we use as the signal to disable GL
        rendering while keeping all selection logic functional.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        self.plotter = QtInteractor(container)
        self.plotter.set_background("white")

        self._picking_enabled = False
        try:
            self.plotter.enable_cell_picking(
                callback=self._handle_cell_pick_callback,
                through=False,
                show_message=True,
                left_clicking=True,
            )
            self._picking_enabled = True
        except RuntimeError:
            # Non-interactive rendering context (offscreen tests / CI).
            self._picking_enabled = False

        layout.addWidget(self.plotter.interactor)
        return container

    # ----------------------------------------------------------------------
    # STEP loading
    # ----------------------------------------------------------------------
    def _on_load_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load STEP File",
            "",
            "STEP Files (*.step *.stp);;All Files (*)",
        )
        if file_path:
            try:
                self.load_step_file(file_path)
            except Exception as exc:  # Surface errors to the user.
                self._set_status(f"Load failed: {exc}")

    def load_step_file(self, step_file: str,
                       mesh_size_min: Optional[float] = None,
                       mesh_size_max: Optional[float] = None) -> MeshData:
        """Load a STEP file, mesh it, render it, and return the MeshData.

        Uses fast, responsive bounding-box relative mesh sizes (computed by
        :class:`CADGeometryPipeline` when ``mesh_size_min`` / ``mesh_size_max``
        are ``None``) so GUI loading stays interactive. The pipeline also
        disables Gmsh's CAD-embedded characteristic-length and curvature
        propagation to prevent millions of tiny elements.
        """
        mesh_data = build_mesh_data(
            step_file,
            mesh_size_min=mesh_size_min,
            mesh_size_max=mesh_size_max,
        )
        self._mesh_data = mesh_data
        self._selected_surface_tag = None
        self._highlight_overlay_triangles = None

        self._render_mesh()
        self._update_cad_info()
        self._clear_selection_details()
        self._set_status(f"Loaded: {os.path.basename(step_file)}")
        return mesh_data

    # ----------------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------------
    def _render_mesh(self) -> None:
        assert self._mesh_data is not None

        mesh = _build_polydata(self._mesh_data)
        self._surface_polydata = mesh

        if not self._picking_enabled:
            # Headless / no-GL context: keep the Polydata for logic
            # verification but avoid GL calls (which abort in
            # vtkWin32OpenGLRenderWindow). Interactive mode renders below.
            self._surface_mesh_actor = None
            return

        self.plotter.clear()
        actor = self.plotter.add_mesh(
            mesh,
            name="cad_surface",
            color="lightblue",
            show_edges=True,
            opacity=1.0,
            pickable=True,
        )
        self._surface_mesh_actor = actor
        self.plotter.reset_camera()

    def _update_cad_info(self) -> None:
        assert self._mesh_data is not None
        md = self._mesh_data
        self.nodes_label.setText(f"Nodes      : {md.n_nodes}")
        self.elements_label.setText(f"TET10 mesh : {md.n_elements}")
        self.faces_label.setText(f"Surf faces : {md.n_boundary_faces}")
        self.tags_label.setText(f"Surf tags  : {len(md.surface_tags)}  {md.surface_tags}")
        self.file_path_label.setText(f"File: {os.path.basename(md.step_file)}")

    def _clear_selection_details(self) -> None:
        self.sel_tag_label.setText("Selected CAD surface : —")
        self.sel_tri_label.setText("Triangle count       : —")
        self.sel_node_label.setText("Node count           : —")

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    # ----------------------------------------------------------------------
    # Cell-picking callback (Approach A)
    # ----------------------------------------------------------------------
    def _handle_cell_pick_callback(self, picked) -> None:
        """PyVista cell-picking callback.

        Different pyvista/pyvistaqt versions pass either the picked mesh or
        the picker object; handle both.
        """
        # Newer pyvista passes a picker with a ``picked_mesh`` attribute.
        if hasattr(picked, "picked_mesh"):
            picked_mesh = getattr(picked, "picked_mesh")
        else:
            picked_mesh = picked

        if picked_mesh is None or picked_mesh.n_cells == 0:
            return

        # Determine the original PolyData cell id.
        if "vtkOriginalCellIds" in picked_mesh.cell_data:
            picked_face_idx = int(
                picked_mesh.cell_data["vtkOriginalCellIds"][0]
            )
        elif "cell_index" in picked_mesh.cell_data:
            picked_face_idx = int(picked_mesh.cell_data["cell_index"][0])
        else:
            # Fallback: use the picked mesh's own cell index.
            picked_face_idx = int(picked_mesh.cell_index[0])

        self.select_surface_for_picked_face(picked_face_idx)

    def select_surface_for_picked_face(self, picked_face_idx: int) -> int:
        """Highlight the full CAD surface containing the picked triangle.

        Returns the selected CAD surface tag.
        """
        assert self._mesh_data is not None, "No mesh loaded"
        md = self._mesh_data

        tag = int(md.boundary_surface_tags[picked_face_idx])
        tris = md.surface_tag_to_triangles[tag]
        nodes_for_tag = md.surface_tag_to_nodes[tag]

        # Build the translucent highlight overlay containing ALL triangles
        # that belong to this CAD surface tag. (Headless mode keeps the
        # Polydata for logic verification and skips GL rendering.)
        overlay = _build_polydata(md, faces=tris)
        self._highlight_overlay = overlay

        if self._picking_enabled:
            if self.HIGHLIGHT_NAME in self.plotter.renderer.actors:
                self.plotter.remove_actor(self.HIGHLIGHT_NAME)

            self.plotter.add_mesh(
                overlay,
                name=self.HIGHLIGHT_NAME,
                color="red",
                opacity=0.5,
                pickable=False,
                show_edges=False,
            )
            self.plotter.render()

        # Update internal state and sidebar.
        self._selected_surface_tag = tag
        self._highlight_overlay_triangles = tris

        self.sel_tag_label.setText(f"Selected CAD surface : {tag}")
        self.sel_tri_label.setText(f"Triangle count       : {len(tris)}")
        self.sel_node_label.setText(f"Node count           : {len(nodes_for_tag)}")
        self._set_status(
            f"Selected CAD surface {tag}: {len(tris)} triangles, "
            f"{len(nodes_for_tag)} nodes"
        )
        return tag

    # ----------------------------------------------------------------------
    # Test-facing API
    # ----------------------------------------------------------------------
    def get_mesh_data(self) -> Optional[MeshData]:
        return self._mesh_data

    def get_surface_tags(self) -> List[int]:
        assert self._mesh_data is not None
        return list(self._mesh_data.surface_tags)

    def get_selected_surface_tag(self) -> Optional[int]:
        return self._selected_surface_tag

    def get_highlight_overlay_triangles(self) -> Optional[np.ndarray]:
        """Return the (M,3) triangle array of the current highlight overlay."""
        return self._highlight_overlay_triangles

    def get_highlight_overlay(self) -> Optional[pv.PolyData]:
        """Return the highlight overlay Polydata (available even headless)."""
        return self._highlight_overlay

    def get_surface_mesh(self) -> Optional[pv.PolyData]:
        """Return the rendered boundary-surface Polydata."""
        return self._surface_polydata

    def is_picking_enabled(self) -> bool:
        """True when interactive cell picking (real GL viewport) is available."""
        return self._picking_enabled

    def get_selected_surface_stats(self) -> Optional[Dict[str, int]]:
        if self._selected_surface_tag is None:
            return None
        assert self._mesh_data is not None
        md = self._mesh_data
        tag = self._selected_surface_tag
        return {
            "tag": tag,
            "n_triangles": len(md.surface_tag_to_triangles[tag]),
            "n_nodes": len(md.surface_tag_to_nodes[tag]),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    app = QApplication.instance() or QApplication([])
    window = FEAAppMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())