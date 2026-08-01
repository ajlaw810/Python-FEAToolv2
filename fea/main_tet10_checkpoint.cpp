// ============================================================================
// main_tet10_checkpoint.cpp
// ----------------------------------------------------------------------------
// Checkpoint 1.1: TET10 element stiffness matrix eigenvalue spectrum test.
//
//   * Builds a single unit TET10 element (corners at (0,0,0), (1,0,0),
//     (0,1,0), (0,0,1); mid-edge nodes at exact edge midpoints).
//   * Material: E = 210 GPa, nu = 0.3 (structural steel).
//   * Computes the 30x30 local stiffness matrix K_e.
//   * Performs a full eigenvalue decomposition via
//     Eigen::SelfAdjointEigenSolver and validates:
//       - EXACTLY 6 eigenvalues are effectively zero (< 1e-3 * lambda_max)
//         -> 6 rigid body modes (3 translations + 3 rotations)
//       - The remaining 24 eigenvalues are strictly positive
//         -> positive definiteness under constrained motion
//
// Build (from the fea/ directory):
//   g++ -std=c++20 -O2 -I<path-to-eigen> main_tet10_checkpoint.cpp -o tet10_checkpoint
//   ./tet10_checkpoint
// ============================================================================

#include "tet10_element.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

// ----------------------------------------------------------------------------
// Build the unit TET10 element.
//   Corners   : 1 (0,0,0), 2 (1,0,0), 3 (0,1,0), 4 (0,0,1)
//   Mid-edges : 5 (1-2), 6 (2-3), 7 (3-1), 8 (1-4), 9 (2-4), 10 (3-4)
// ----------------------------------------------------------------------------
fea::Tet10Element::CoordMatrix makeUnitTet10() {
    fea::Tet10Element::CoordMatrix nodes;
    // Corner nodes
    nodes.row(0) << 0.0, 0.0, 0.0;  // node 1
    nodes.row(1) << 1.0, 0.0, 0.0;  // node 2
    nodes.row(2) << 0.0, 1.0, 0.0;  // node 3
    nodes.row(3) << 0.0, 0.0, 1.0;  // node 4
    // Mid-edge nodes (exact midpoints)
    nodes.row(4) << 0.5, 0.0, 0.0;  // node 5: edge 1-2
    nodes.row(5) << 0.5, 0.5, 0.0;  // node 6: edge 2-3
    nodes.row(6) << 0.0, 0.5, 0.0;  // node 7: edge 3-1
    nodes.row(7) << 0.0, 0.0, 0.5;  // node 8: edge 1-4
    nodes.row(8) << 0.5, 0.0, 0.5;  // node 9: edge 2-4
    nodes.row(9) << 0.0, 0.5, 0.5;  // node 10: edge 3-4
    return nodes;
}

// ----------------------------------------------------------------------------
// Run Checkpoint 1.1 and return EXIT_SUCCESS / EXIT_FAILURE.
// ----------------------------------------------------------------------------
int runCheckpoint11() {
    // --- Material properties (structural steel) ------------------------------
    constexpr double E = 210.0e9;   // Young's modulus [Pa]
    constexpr double nu = 0.3;      // Poisson's ratio [-]

    // --- Build element and stiffness matrix ----------------------------------
    const fea::Tet10Element element(makeUnitTet10());
    const fea::Tet10Element::StiffnessMatrix Ke =
        element.computeStiffnessMatrix(E, nu);

    std::cout << "============================================================\n";
    std::cout << " TET10 Checkpoint 1.1 - Eigenvalue Spectrum Test\n";
    std::cout << "============================================================\n";
    std::cout << " Material      : E = " << E << " Pa, nu = " << nu << "\n";
    std::cout << " Element volume: " << std::setprecision(12)
              << element.computeVolume() << " (exact = 1/6)\n";
    std::cout << " Stiffness K_e : " << Ke.rows() << " x " << Ke.cols() << "\n";
    std::cout << " Max |K_ij|    : " << std::scientific << std::setprecision(6)
              << Ke.cwiseAbs().maxCoeff() << "\n\n";

    // --- Eigenvalue decomposition --------------------------------------------
    Eigen::SelfAdjointEigenSolver<fea::Tet10Element::StiffnessMatrix> solver(Ke);
    if (solver.info() != Eigen::Success) {
        std::cerr << "[ERROR] Eigen decomposition failed.\n";
        return EXIT_FAILURE;
    }

    // --- Sort eigenvalues ascending ------------------------------------------
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    std::vector<double> lambda(eigenvalues.data(),
                               eigenvalues.data() + eigenvalues.size());
    std::sort(lambda.begin(), lambda.end());

    const double lambda_max = lambda.back();
    const double tol = 1e-3 * lambda_max;  // relative rigid-body threshold

    // --- Print all 30 eigenvalues --------------------------------------------
    std::cout << " Eigenvalues of K_e (sorted ascending):\n";
    std::cout << " --------------------------------------\n";
    for (std::size_t i = 0; i < lambda.size(); ++i) {
        std::cout << "  lambda[" << std::setw(2) << i << "] = "
                  << std::scientific << std::setprecision(8)
                  << lambda[i] << "\n";
    }
    std::cout << "\n";

    // --- Count rigid body modes (effectively zero eigenvalues) ---------------
    int num_zero = 0;
    for (double l : lambda) {
        if (std::abs(l) < tol) ++num_zero;
    }

    // --- Verify remaining eigenvalues are strictly positive ------------------
    bool all_positive = true;
    for (std::size_t i = static_cast<std::size_t>(num_zero); i < lambda.size(); ++i) {
        if (!(lambda[i] > 0.0)) {
            all_positive = false;
            break;
        }
    }

    // --- Report ---------------------------------------------------------------
    std::cout << "============================================================\n";
    std::cout << " Checkpoint 1.1 Results\n";
    std::cout << "============================================================\n";
    std::cout << "  lambda_max            = " << std::scientific
              << std::setprecision(8) << lambda_max << "\n";
    std::cout << "  rigid-body threshold  = " << std::scientific
              << std::setprecision(8) << tol << "\n";
    std::cout << "  effectively-zero evals= " << num_zero
              << " (expected 6: 3 translations + 3 rotations)\n";
    std::cout << "  strictly-positive evals= " << (30 - num_zero)
              << " (expected 24)\n\n";

    // --- Automated assertions -------------------------------------------------
    bool passed = true;

    if (num_zero != 6) {
        std::cerr << "[FAIL] Expected exactly 6 rigid body modes, found "
                  << num_zero << ".\n";
        passed = false;
    }

    if (!all_positive) {
        std::cerr << "[FAIL] Found non-positive eigenvalues among the "
                  << (30 - num_zero) << " non-rigid-body modes.\n";
        passed = false;
    }

    if (passed) {
        std::cout << "[CHECKPOINT 1.1 PASSED]: Rigid Body Modes and Positive "
                     "Definiteness Verified!\n";
        return EXIT_SUCCESS;
    }

    std::cerr << "[CHECKPOINT 1.1 FAILED]\n";
    return EXIT_FAILURE;
}

}  // namespace

int main() {
    return runCheckpoint11();
}