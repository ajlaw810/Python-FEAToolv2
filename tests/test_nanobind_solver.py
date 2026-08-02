import time
import numpy as np
import pytest

from cpp_python_project.core import FEASolver


def build_cantilever_beam_mesh(
    length=10.0, width=1.0, height=1.0, sub_len=20, sub_w=2, sub_h=2
):
    """Generates a structured TET10 mesh for a cantilever beam."""
    ncx = sub_len + 1
    ncy = sub_w + 1
    ncz = sub_h + 1
    n_corners = ncx * ncy * ncz

    dx = length / sub_len
    dy = width / sub_w
    dz = height / sub_h

    corners = np.zeros((n_corners, 3), dtype=np.float64)

    def corner_index(i, j, k):
        return i + j * ncx + k * ncx * ncy

    for k in range(ncz):
        for j in range(ncy):
            for i in range(ncx):
                idx = corner_index(i, j, k)
                corners[idx] = [i * dx, j * dy, k * dz]

    node_pts = [list(c) for c in corners]
    mid_node = {}

    def get_or_create_mid(a, b):
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in mid_node:
            return mid_node[key]
        idx = len(node_pts)
        mid_pt = [
            (corners[a][0] + corners[b][0]) * 0.5,
            (corners[a][1] + corners[b][1]) * 0.5,
            (corners[a][2] + corners[b][2]) * 0.5,
        ]
        node_pts.append(mid_pt)
        mid_node[key] = idx
        return idx

    tet_corners_hex = [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ]

    elements = []

    for k in range(sub_h):
        for j in range(sub_w):
            for i in range(sub_len):
                h = [
                    corner_index(i, j, k),
                    corner_index(i + 1, j, k),
                    corner_index(i + 1, j + 1, k),
                    corner_index(i, j + 1, k),
                    corner_index(i, j, k + 1),
                    corner_index(i + 1, j, k + 1),
                    corner_index(i + 1, j + 1, k + 1),
                    corner_index(i, j + 1, k + 1),
                ]
                for tet in tet_corners_hex:
                    c0, c1, c2, c3 = (
                        h[tet[0]],
                        h[tet[1]],
                        h[tet[2]],
                        h[tet[3]],
                    )
                    elem_nodes = [
                        c0,
                        c1,
                        c2,
                        c3,
                        get_or_create_mid(c0, c1),
                        get_or_create_mid(c1, c2),
                        get_or_create_mid(c2, c0),
                        get_or_create_mid(c0, c3),
                        get_or_create_mid(c1, c3),
                        get_or_create_mid(c2, c3),
                    ]
                    elements.append(elem_nodes)

    nodes_arr = np.array(node_pts, dtype=np.float64)
    elems_arr = np.array(elements, dtype=np.int32)
    return nodes_arr, elems_arr


def test_nanobind_solver_cantilever():
    nodes, elements = build_cantilever_beam_mesh(
        length=10.0, width=1.0, height=1.0, sub_len=20, sub_w=2, sub_h=2
    )

    E = 210.0e9  # 210 GPa
    nu = 0.3

    t0 = time.perf_counter()
    solver = FEASolver(nodes, elements, E, nu)
    t1 = time.perf_counter()
    array_passing_overhead_ms = (t1 - t0) * 1000.0

    print(f"Array passing overhead: {array_passing_overhead_ms:.4f} ms")
    assert array_passing_overhead_ms < 1.0, (
        f"Array passing overhead ({array_passing_overhead_ms:.2f} ms) exceeds 1 ms!"
    )

    fixed_node_ids = np.where(np.abs(nodes[:, 0] - 0.0) < 1e-9)[0].astype(
        np.int32
    )
    tip_node_ids = np.where(np.abs(nodes[:, 0] - 10.0) < 1e-9)[0].astype(
        np.int32
    )

    total_force_z = -100000.0  # -100 kN
    f_per_node = total_force_z / len(tip_node_ids)
    forces = np.zeros((len(tip_node_ids), 3), dtype=np.float64)
    forces[:, 2] = f_per_node

    solver.apply_fixed_bc(fixed_node_ids)
    solver.apply_point_loads(tip_node_ids, forces)

    solver.solve()

    displacements = solver.get_displacements()
    assert displacements.shape == (len(nodes), 3)

    tip_uz = displacements[tip_node_ids, 2]
    max_tip_deflection = np.max(np.abs(tip_uz))

    I = 1.0 * (1.0**3) / 12.0
    delta_exact = 100000.0 * (10.0**3) / (3.0 * E * I)

    rel_error = abs(max_tip_deflection - delta_exact) / delta_exact

    print(f"Max Tip Deflection (FEA) : {max_tip_deflection * 1000.0:.4f} mm")
    print(f"Exact Tip Deflection (Euler-Bernoulli): {delta_exact * 1000.0:.4f} mm")
    print(f"Relative Error           : {rel_error * 100.0:.2f}%")

    assert rel_error < 0.03, f"Relative error {rel_error * 100:.2f}% exceeds 3% tolerance!"

    vm_stresses = solver.get_von_mises_stresses()
    assert vm_stresses.shape == (len(nodes),)

    max_vm_stress = np.max(vm_stresses)
    max_vm_node_idx = np.argmax(vm_stresses)
    max_vm_node_x = nodes[max_vm_node_idx, 0]

    print(f"Max Von Mises Stress     : {max_vm_stress / 1e6:.2f} MPa at X = {max_vm_node_x:.2f} m")

    assert abs(max_vm_node_x - 0.0) < 0.5, (
        f"Peak stress expected at root X = 0, but found at X = {max_vm_node_x}"
    )

    print("\n[CHECKPOINT 2.1 PASSED]: Nanobind Zero-Copy Bridge & Nodal Stress Recovery Verified!\n")
