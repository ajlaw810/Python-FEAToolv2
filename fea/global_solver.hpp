#ifndef FEA_GLOBAL_SOLVER_HPP
#define FEA_GLOBAL_SOLVER_HPP

// ============================================================================
// global_solver.hpp
// ----------------------------------------------------------------------------
// Global sparse stiffness assembly + boundary conditions + linear solve for
// 3D structural FEA with TET10 elements (Checkpoint 1.2).
//
//   * Assembles the global stiffness matrix K (size 3*N_nodes x 3*N_nodes)
//     using Eigen::SparseMatrix<double> from Eigen::Triplet<double>.
//   * Applies fixed-displacement constraints (u_x = u_y = u_z = 0) by
//     rebuilding K from filtered triplets: constrained rows/columns are
//     zeroed, K_ii = 1.0, and the corresponding F_i entries are zeroed,
//     preserving symmetry.
//   * Applies nodal point loads to the global force vector F.
//   * Solves  K * u = F  with Eigen::SimplicialLDLT and verifies the
//     factorization / solve converged (solver.info() == Eigen::Success).
//
// Node ID convention:
//   Global node IDs passed to this class are 0-based (C++ indexing).
//   The TET10 connectivity matrix uses 1-based node IDs
//   (FEA-style numbering) and is converted internally.
//
// Dependencies: Eigen3 (header-only), fea::Tet10Element
// ============================================================================

#include "tet10_element.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <array>
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fea {

class GlobalSolver {
public:
    using SparseMatrix = Eigen::SparseMatrix<double>;
    using Triplet      = Eigen::Triplet<double>;
    using Vector       = Eigen::VectorXd;
    using Matrix       = Eigen::MatrixXd;
    using IndexVector  = Eigen::VectorXi;

    // ------------------------------------------------------------------------
    // Constructor.
    //
    //   nodes : (N_nodes x 3) global nodal coordinates, row r = [x, y, z].
    //   conn  : (N_elements x 10) TET10 connectivity matrix, 1-based node IDs.
    //           Local node order follows the standard TET10 convention:
    //           corners 1-4, mid-edges 5 (1-2), 6 (2-3), 7 (3-1),
    //           8 (1-4), 9 (2-4), 10 (3-4).
    //   E, nu : isotropic material properties.
    //
    // Throws std::invalid_argument on inconsistent dimensions or invalid
    // (out-of-range) connectivity.
    // ------------------------------------------------------------------------
    GlobalSolver(const Matrix& nodes, const Eigen::MatrixXi& conn, double E, double nu)
        : n_nodes_(static_cast<int>(nodes.rows())),
          n_dofs_(3 * n_nodes_),
          E_(E),
          nu_(nu) {
        if (nodes.cols() != 3) {
            throw std::invalid_argument(
                "GlobalSolver: node coordinate matrix must have 3 columns "
                "(x, y, z).");
        }
        if (conn.cols() != Tet10Element::kNumNodes) {
            throw std::invalid_argument(
                "GlobalSolver: connectivity matrix must have 10 columns "
                "(N_elements x 10).");
        }
        const int n_elems = static_cast<int>(conn.rows());

        K_.resize(n_dofs_, n_dofs_);
        F_ = Vector::Zero(n_dofs_);
        u_ = Vector::Zero(n_dofs_);
        constrained_.assign(n_dofs_, false);

        assemble_(nodes, conn, n_elems);
    }

    // ------------------------------------------------------------------------
    // Apply fixed-displacement constraints (u_x = u_y = u_z = 0) to a list of
    // node IDs (0-based). Constrained DOFs get row/col zeroed, K_ii = 1.0 and
    // F_i = 0.0, preserving symmetry. May be called multiple times.
    // ------------------------------------------------------------------------
    void applyFixedConstraints(const IndexVector& node_ids) {
        bool any_new = false;
        for (int i = 0; i < node_ids.size(); ++i) {
            const int node = node_ids(i);
            checkNodeId(node);
            for (int d = 0; d < 3; ++d) {
                const int dof = 3 * node + d;
                if (!constrained_[static_cast<std::size_t>(dof)]) {
                    constrained_[static_cast<std::size_t>(dof)] = true;
                    F_(dof) = 0.0;
                    any_new = true;
                }
            }
        }
        if (any_new) {
            rebuildK_();
        }
    }

    // Single-node overload.
    void applyFixedConstraints(int node_id) {
        checkNodeId(node_id);
        for (int d = 0; d < 3; ++d) {
            const int dof = 3 * node_id + d;
            if (!constrained_[static_cast<std::size_t>(dof)]) {
                constrained_[static_cast<std::size_t>(dof)] = true;
                F_(dof) = 0.0;
            }
        }
        rebuildK_();
    }

    // ------------------------------------------------------------------------
    // Apply point loads at specified node IDs (0-based). forces is a
    // (N x 3) matrix, row i = [fx, fy, fz] added to node node_ids(i).
    // ------------------------------------------------------------------------
    void applyPointLoads(const IndexVector& node_ids, const Matrix& forces) {
        if (node_ids.size() != forces.rows() || forces.cols() != 3) {
            throw std::invalid_argument(
                "GlobalSolver::applyPointLoads: node_ids.size() must equal "
                "forces.rows() and forces must have 3 columns.");
        }
        for (int i = 0; i < node_ids.size(); ++i) {
            const int node = node_ids(i);
            checkNodeId(node);
            for (int d = 0; d < 3; ++d) {
                F_(3 * node + d) += forces(i, d);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Solve K * u = F with Eigen::SimplicialLDLT.
    // Returns the displacement vector u (size 3*N_nodes).
    // Throws std::runtime_error if the factorization or solve fails.
    // ------------------------------------------------------------------------
    const Vector& solve() {
        // 1. Check for unconstrained system (rigid body modes)
        const bool has_constraints = std::any_of(
            constrained_.begin(), constrained_.end(), [](bool c) { return c; });
        if (!has_constraints) {
            throw std::runtime_error(
                "System is unconstrained! Rigid body modes present.");
        }

        Eigen::SimplicialLDLT<SparseMatrix> solver;
        solver.compute(K_);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "GlobalSolver::solve: SimplicialLDLT factorization failed "
                "(matrix may be singular/indefinite).");
        }
        u_ = solver.solve(F_);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "GlobalSolver::solve: SimplicialLDLT back-substitution failed.");
        }

        // 2. Check for NaN values or explosive displacement magnitudes
        if (u_.hasNaN() || u_.cwiseAbs().maxCoeff() > 1e10) {
            throw std::runtime_error(
                "Numerical instability or unconstrained system detected: "
                "solution contains NaN or non-physical displacement (> 1e10).");
        }

        return u_;
    }


    // --- Accessors ------------------------------------------------------------
    const SparseMatrix& K() const { return K_; }
    const Vector&       F() const { return F_; }
    const Vector&       u() const { return u_; }
    int   nNodes() const { return n_nodes_; }
    int   nDofs() const { return n_dofs_; }

    // --- Small stats report for benchmark logging -----------------------------
    void printStats(std::ostream& os) const {
        os << " Global system      : " << n_dofs_ << " x " << n_dofs_
           << " DOFs\n";
        os << " Non-zeros in K     : " << K_.nonZeros() << "\n";
        os << " Constrained DOFs   : "
           << static_cast<int>(std::count(constrained_.begin(),
                                          constrained_.end(), true))
           << "\n";
    }

private:
    // ------------------------------------------------------------------------
    // Element loop: compute each 30x30 Ke and scatter to triplet list.
    // ------------------------------------------------------------------------
    void assemble_(const Matrix& nodes, const Eigen::MatrixXi& conn, int n_elems) {
        triplets_.clear();
        // Up to 900 non-zero entries per element.
        triplets_.reserve(static_cast<std::size_t>(n_elems) *
                          Tet10Element::kNumDofs * Tet10Element::kNumDofs);

        for (int e = 0; e < n_elems; ++e) {
            // Extract the 10 global node IDs (1-based -> 0-based).
            std::array<int, Tet10Element::kNumNodes> elem_nodes{};
            for (int a = 0; a < Tet10Element::kNumNodes; ++a) {
                elem_nodes[static_cast<std::size_t>(a)] =
                    conn(e, a) - 1;  // convert to 0-based
                checkNodeId(elem_nodes[static_cast<std::size_t>(a)]);
            }

            // Extract 10x3 local coordinates.
            Tet10Element::CoordMatrix local;
            for (int a = 0; a < Tet10Element::kNumNodes; ++a) {
                local.row(a) = nodes.row(elem_nodes[static_cast<std::size_t>(a)]);
            }

            // Local stiffness.
            const Tet10Element element(local);
            const Tet10Element::StiffnessMatrix Ke =
                element.computeStiffnessMatrix(E_, nu_);

            // Scatter 30x30 Ke into global triplets.
            for (int i = 0; i < Tet10Element::kNumDofs; ++i) {
                const int global_i =
                    3 * elem_nodes[static_cast<std::size_t>(i / 3)] + (i % 3);
                for (int j = 0; j < Tet10Element::kNumDofs; ++j) {
                    const int global_j =
                        3 * elem_nodes[static_cast<std::size_t>(j / 3)] + (j % 3);
                    triplets_.emplace_back(global_i, global_j, Ke(i, j));
                }
            }
        }

        // Build the initial assembled matrix.
        rebuildK_();
    }

    // ------------------------------------------------------------------------
    // Rebuild K from the stored triplets, dropping any triplet that touches a
    // constrained DOF and inserting unit diagonals for constrained DOFs.
    // This exactly implements "set row/col to 0, K_ii = 1.0, F_i = 0.0" and
    // preserves symmetry without fragile in-place CSR surgery.
    // ------------------------------------------------------------------------
    void rebuildK_() {
        std::vector<Triplet> filtered;
        filtered.reserve(triplets_.size());

        for (const Triplet& t : triplets_) {
            if (!constrained_[static_cast<std::size_t>(t.row())] &&
                !constrained_[static_cast<std::size_t>(t.col())]) {
                filtered.emplace_back(t);
            }
        }

        // Unit diagonal for each constrained DOF (K_ii = 1.0).
        for (int i = 0; i < n_dofs_; ++i) {
            if (constrained_[static_cast<std::size_t>(i)]) {
                filtered.emplace_back(i, i, 1.0);
            }
        }

        K_.resize(n_dofs_, n_dofs_);
        K_.setFromTriplets(filtered.begin(), filtered.end());
        K_.makeCompressed();

        // F_i = 0.0 for constrained DOFs is enforced in applyFixedConstraints;
        // ensure it holds here too in case of repeated calls.
        for (int i = 0; i < n_dofs_; ++i) {
            if (constrained_[static_cast<std::size_t>(i)]) {
                F_(i) = 0.0;
            }
        }
    }

    void checkNodeId(int node_id) const {
        if (node_id < 0 || node_id >= n_nodes_) {
            throw std::invalid_argument(
                "GlobalSolver: node ID " + std::to_string(node_id) +
                " out of range [0, " + std::to_string(n_nodes_ - 1) + "].");
        }
    }

    int n_nodes_;
    int n_dofs_;
    double E_;
    double nu_;

    SparseMatrix K_;
    Vector F_;
    Vector u_;

    std::vector<bool> constrained_;  // per-DOF fixed-constraint flag
    std::vector<Triplet> triplets_;  // original (full) triplet list
};

}  // namespace fea

#endif  // FEA_GLOBAL_SOLVER_HPP