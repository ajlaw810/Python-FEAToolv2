// ============================================================================
// main_beam_checkpoint.cpp
// ----------------------------------------------------------------------------
// Checkpoint 1.2: Global Sparse Assembly + Cantilever Beam Theoretical
// Benchmark (Euler-Bernoulli tip-load deflection).
//
//   * Builds a structured beam mesh (L x w x h = 10 x 1 x 1 m) as a grid of
//     sub-cubes. Each sub-cube is split into 6 TET10 elements using the
//     Freudenthal/Kuhn triangulation around the body diagonal 0-6.
//   * TET10 mid-edge nodes are created on demand via a sorted-pair (edge)
//     map keyed on the global corner IDs, so every tetrahedral edge
//     (hex edge, hex face diagonal, or hex body diagonal) has a unique
//     global midpoint node shared consistently across neighbouring
//     elements/hexes.
//   * Material: E = 210 GPa, nu = 0.3 (structural steel).
//   * Boundary conditions: fully clamp all nodes on the face X = 0.
//   * Loading: total F_z = -100,000 N distributed evenly over all nodes on
//     the tip face X = L.
//   * Assembles the global sparse K with fea::GlobalSolver, applies the BCs
//     (row/col zeroing, K_ii = 1, F_i = 0 -> symmetric modification),
//     solves K*u = F with Eigen::SimplicialLDLT and verifies convergence.
//   * Compares the numerical tip deflection delta_FEA to the exact
//     Euler-Bernoulli tip deflection
//           I = w*h^3/12,  delta_exact = F*L^3/(3*E*I)
//     and asserts the relative error is below 3.0%.
//
// Build (from the fea/ directory):
//   g++ -std=c++20 -O2 -I<path-to-eigen> main_beam_checkpoint.cpp -o beam_checkpoint
//   ./beam_checkpoint
// ============================================================================

#include "global_solver.hpp"
#include "tet10_element.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

namespace {

// ----------------------------------------------------------------------------
// Benchmark parameters.
// ----------------------------------------------------------------------------
constexpr double kLength   = 10.0;       // beam length along x  [m]
constexpr double kWidth    = 1.0;        // beam width  along y  [m]
constexpr double kHeight   = 1.0;        // beam height along z  [m]
constexpr double kE        = 210.0e9;    // Young's modulus       [Pa]
constexpr double kNu       = 0.3;        // Poisson's ratio       [-]
constexpr double kTotalF   = 100000.0;   // total downward tip force magnitude [N]
constexpr double kRelErrTol = 0.03;      // 3% relative-error tolerance

// ----------------------------------------------------------------------------
// Structured mesh. 20 x 2 x 2 sub-cubes -> 480 TET10 elements.
// ----------------------------------------------------------------------------
constexpr int kSubLen = 20;              // sub-cubes along length (x)
constexpr int kSubW   = 2;               // sub-cubes across width  (y)
constexpr int kSubH   = 2;               // sub-cubes across height (z)

constexpr int kNumHexes = kSubLen * kSubW * kSubH;          // 80 hexes
constexpr int kTetsPerHex = 6;                              // Kuhn split
constexpr int kNumElements = kNumHexes * kTetsPerHex;       // 480 elements

// ----------------------------------------------------------------------------
// Structured-corner index (global corner-node id for a grid point).
// ----------------------------------------------------------------------------
inline int cornerIndex(int i, int j, int k) {
    return i + j * (kSubLen + 1) + k * (kSubLen + 1) * (kSubW + 1);
}

// Hex -> 6 TET10 corner sets sharing the body diagonal 0-6
// (Freudenthal / Kuhn triangulation). Each tet has volume 1/6 of the hex.
constexpr int kTetCorners[kTetsPerHex][4] = {
    {0, 1, 2, 6},
    {0, 2, 3, 6},
    {0, 3, 7, 6},
    {0, 7, 4, 6},
    {0, 4, 5, 6},
    {0, 5, 1, 6},
};

struct BeamMesh {
    Eigen::MatrixXd nodes;  // (N_nodes x 3) 0-based rows = [x, y, z]
    Eigen::MatrixXi conn;   // (N_elements x 10) TET10 connectivity, 1-based
};

// ----------------------------------------------------------------------------
// Build the structured hex mesh and convert it to TET10 elements.
//
// Every tetrahedron edge of the Kuhn triangulation connects two hex corners,
// so a single map keyed on the sorted pair of global corner IDs yields a
// unique global midpoint node per physical segment. This automatically:
//   * deduplicates mid-edge nodes shared between neighbouring tets,
//   * bumps mid-edge nodes on hex faces shared with neighbouring hexes
//     (face diagonals are geometrically identical across the face), and
//   * places every TET10 mid-edge node at the exact edge midpoint.
// ----------------------------------------------------------------------------
BeamMesh buildBeamMesh() {
    const int ncx = kSubLen + 1;
    const int ncy = kSubW + 1;
    const int ncz = kSubH + 1;
    const int n_corners = ncx * ncy * ncz;

    const double dx = kLength / static_cast<double>(kSubLen);
    const double dy = kWidth / static_cast<double>(kSubW);
    const double dz = kHeight / static_cast<double>(kSubH);

    // --- Corner-node coordinates ---------------------------------------------
    Eigen::MatrixXd corners(n_corners, 3);
    for (int k = 0; k < ncz; ++k) {
        for (int j = 0; j < ncy; ++j) {
            for (int i = 0; i < ncx; ++i) {
                corners.row(cornerIndex(i, j, k)) <<
                    static_cast<double>(i) * dx,
                    static_cast<double>(j) * dy,
                    static_cast<double>(k) * dz;
            }
        }
    }

    // --- Node accumulator (corner ids first = ids 0..n_corners-1) -----------
    std::vector<std::array<double, 3>> node_pts;
    node_pts.reserve(static_cast<std::size_t>(n_corners) * 6);
    for (int c = 0; c < n_corners; ++c) {
        node_pts.push_back({corners(c, 0), corners(c, 1), corners(c, 2)});
    }

    // --- Mid-edge node map (sorted pair of corner ids -> node id) ------------
    std::map<std::pair<int, int>, int> mid_node;
    auto getOrCreateMid = [&](int a, int b) -> int {
        if (a > b) std::swap(a, b);
        const auto key = std::make_pair(a, b);
        const auto it = mid_node.find(key);
        if (it != mid_node.end()) {
            return it->second;
        }
        const int id = static_cast<int>(node_pts.size());
        node_pts.push_back({
            (corners(a, 0) + corners(b, 0)) * 0.5,
            (corners(a, 1) + corners(b, 1)) * 0.5,
            (corners(a, 2) + corners(b, 2)) * 0.5});
        mid_node.emplace(key, id);
        return id;
    };

    // --- Element loop ----------------------------------------------------------
    std::vector<std::array<int, 10>> elems;
    elems.reserve(static_cast<std::size_t>(kNumElements));

    for (int k = 0; k < kSubH; ++k) {
        for (int j = 0; j < kSubW; ++j) {
            for (int i = 0; i < kSubLen; ++i) {
                // Global corner IDs of the current hex (local order 0..7).
                const std::array<int, 8> h = {{
                    cornerIndex(i,     j,     k),
                    cornerIndex(i + 1, j,     k),
                    cornerIndex(i + 1, j + 1, k),
                    cornerIndex(i,     j + 1, k),
                    cornerIndex(i,     j,     k + 1),
                    cornerIndex(i + 1, j,     k + 1),
                    cornerIndex(i + 1, j + 1, k + 1),
                    cornerIndex(i,     j + 1, k + 1),
                }};

                for (const auto& tet : kTetCorners) {
                    const int c0 = h[static_cast<std::size_t>(tet[0])];
                    const int c1 = h[static_cast<std::size_t>(tet[1])];
                    const int c2 = h[static_cast<std::size_t>(tet[2])];
                    const int c3 = h[static_cast<std::size_t>(tet[3])];

                    // TET10 node order: corners 1-4, then mid-edges
                    // 5 (1-2), 6 (2-3), 7 (3-1), 8 (1-4), 9 (2-4), 10 (3-4).
                    elems.push_back({
                        c0, c1, c2, c3,
                        getOrCreateMid(c0, c1),
                        getOrCreateMid(c1, c2),
                        getOrCreateMid(c2, c0),
                        getOrCreateMid(c0, c3),
                        getOrCreateMid(c1, c3),
                        getOrCreateMid(c2, c3),
                    });
                }
            }
        }
    }

    // --- Convert to Eigen matrices --------------------------------------------
    BeamMesh mesh;
    mesh.nodes.resize(static_cast<int>(node_pts.size()), 3);
    for (std::size_t r = 0; r < node_pts.size(); ++r) {
        mesh.nodes.row(static_cast<int>(r)) <<
            node_pts[r][0], node_pts[r][1], node_pts[r][2];
    }

    mesh.conn.resize(static_cast<int>(elems.size()), 10);
    for (std::size_t e = 0; e < elems.size(); ++e) {
        for (int a = 0; a < 10; ++a) {
            mesh.conn(static_cast<int>(e), a) = elems[e][a] + 1;  // 1-based
        }
    }

    return mesh;
}

// ----------------------------------------------------------------------------
// Collect node IDs (0-based) whose x-coordinate lies on the given face.
// ----------------------------------------------------------------------------
fea::GlobalSolver::IndexVector nodesOnFace(const BeamMesh& mesh, double x_face) {
    constexpr double tol = 1e-9;
    std::vector<int> ids;
    for (int n = 0; n < mesh.nodes.rows(); ++n) {
        if (std::abs(mesh.nodes(n, 0) - x_face) < tol) {
            ids.push_back(n);
        }
    }
    fea::GlobalSolver::IndexVector out(static_cast<int>(ids.size()));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        out(static_cast<int>(i)) = ids[i];
    }
    return out;
}

// ----------------------------------------------------------------------------
// Run Checkpoint 1.2 and return EXIT_SUCCESS / EXIT_FAILURE.
// ----------------------------------------------------------------------------
int runCheckpoint12() {
    std::cout << "============================================================\n";
    std::cout << " TET10 Checkpoint 1.2 - Cantilever Beam Benchmark\n";
    std::cout << "============================================================\n";
    std::cout << " Geometry     : L = " << kLength << " m, w = " << kWidth
              << " m, h = " << kHeight << " m\n";
    std::cout << " Material     : E = " << kE << " Pa, nu = " << kNu << "\n";

    // --- Mesh -----------------------------------------------------------------
    const BeamMesh mesh = buildBeamMesh();
    const int n_nodes = mesh.nodes.rows();
    const int n_elems = mesh.conn.rows();
    std::cout << " Mesh         : " << kSubLen << " x " << kSubW << " x "
              << kSubH << " sub-cubes, " << kTetsPerHex
              << " TET10 per hex (Kuhn split)\n";
    std::cout << " Nodes        : " << n_nodes << "\n";
    std::cout << " TET10 elems  : " << n_elems << "\n";

    // --- Boundary conditions ----------------------------------------------------
    const fea::GlobalSolver::IndexVector fixed_ids = nodesOnFace(mesh, 0.0);
    const fea::GlobalSolver::IndexVector tip_ids = nodesOnFace(mesh, kLength);
    std::cout << " Fixed nodes  : " << fixed_ids.size()
              << " (all DOFs on X = 0)\n";
    std::cout << " Tip nodes    : " << tip_ids.size()
              << " (F_z distributed on X = L)\n\n";

    // --- Assemble + apply BCs + solve ------------------------------------------
    fea::GlobalSolver solver(mesh.nodes, mesh.conn, kE, kNu);
    solver.applyFixedConstraints(fixed_ids);

    // Total downward force split evenly over all tip-face nodes.
    const double f_per_node = -kTotalF / static_cast<double>(tip_ids.size());
    Eigen::MatrixXd forces(tip_ids.size(), 3);
    forces.setZero();
    forces.col(2).setConstant(f_per_node);

    solver.applyPointLoads(tip_ids, forces);

    std::cout << " Assembled global system:\n";
    solver.printStats(std::cout);

    const Eigen::VectorXd& u = solver.solve();
    std::cout << " Solve        : SimplicialLDLT converged (Eigen::Success)\n\n";

    // --- Post-processing ----------------------------------------------------------
    double delta_fea = 0.0;
    double min_uz = 0.0;
    for (int i = 0; i < tip_ids.size(); ++i) {
        const double uz = u(3 * tip_ids(i) + 2);
        min_uz = std::min(min_uz, uz);
        delta_fea = std::max(delta_fea, std::abs(uz));
    }

    // Exact Euler-Bernoulli tip deflection under an end shear load.
    const double I = kWidth * kHeight * kHeight * kHeight / 12.0;  // 1/12 m^4
    const double delta_exact =
        kTotalF * kLength * kLength * kLength / (3.0 * kE * I);

    const double rel_err = std::abs(delta_fea - delta_exact) / delta_exact;

    // --- Report ---------------------------------------------------------------------
    std::cout << "============================================================\n";
    std::cout << " Checkpoint 1.2 Results\n";
    std::cout << "============================================================\n";
    std::cout << "  I             = " << std::fixed << std::setprecision(8)
              << I << " m^4  (w*h^3/12)\n";
    std::cout << "  delta_exact   = " << std::scientific << std::setprecision(8)
              << delta_exact << " m  (F*L^3/(3*E*I))\n";
    std::cout << "  delta_FEA     = " << std::scientific << std::setprecision(8)
              << delta_fea << " m  (max |u_z| at tip)\n";
    std::cout << "  min u_z(tip)  = " << std::scientific << std::setprecision(8)
              << min_uz << " m  (downward = negative)\n";
    std::cout << "  relative error= " << std::fixed << std::setprecision(4)
              << 100.0 * rel_err << "%  (tolerance "
              << 100.0 * kRelErrTol << "%)\n\n";

    // --- Automated assertions ---------------------------------------------------------
    bool passed = true;
    if (!(rel_err < kRelErrTol)) {
        std::cerr << "[FAIL] Relative error " << 100.0 * rel_err
                  << "% exceeds tolerance " << 100.0 * kRelErrTol << "%.\n";
        passed = false;
    }
    if (!(delta_fea > 0.0)) {
        std::cerr << "[FAIL] Numerical tip deflection is not positive.\n";
        passed = false;
    }

    if (passed) {
        std::cout << "[CHECKPOINT 1.2 PASSED]: Cantilever Beam Deflection "
                     "Verified against Beam Theory!\n";
        return EXIT_SUCCESS;
    }

    std::cerr << "[CHECKPOINT 1.2 FAILED]\n";
    return EXIT_FAILURE;
}

}  // namespace

int main() {
    return runCheckpoint12();
}