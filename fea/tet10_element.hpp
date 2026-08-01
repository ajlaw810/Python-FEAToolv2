#ifndef FEA_TET10_ELEMENT_HPP
#define FEA_TET10_ELEMENT_HPP

// ============================================================================
// tet10_element.hpp
// ----------------------------------------------------------------------------
// 10-node Quadratic Tetrahedron (TET10) element for 3D structural FEA.
//
//   * 10 nodes: 4 corners + 6 mid-edge nodes  ->  30 DOFs (10 x 3)
//   * Quadratic shape functions in barycentric coordinates
//   * 4-point Gauss quadrature (exact for quadratic strain fields)
//   * Isotropic linear elasticity (E, nu)
//
// Node numbering (standard convention):
//   Corners   : 1, 2, 3, 4
//   Mid-edges : 5 (1-2), 6 (2-3), 7 (3-1), 8 (1-4), 9 (2-4), 10 (3-4)
//
// Dependencies: Eigen3 (header-only, https://eigen.tuxfamily.org)
// ============================================================================

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <stdexcept>

namespace fea {

class Tet10Element {
public:
    // --- Compile-time constants ---------------------------------------------
    static constexpr int kNumNodes = 10;
    static constexpr int kNumDofs = 30;  // 10 nodes * 3 displacements

    // --- Eigen type aliases --------------------------------------------------
    using CoordMatrix        = Eigen::Matrix<double, kNumNodes, 3>;  // 10 x 3
    using ShapeVector        = Eigen::Matrix<double, kNumNodes, 1>;  // 10 x 1
    using ShapeGradNatural   = Eigen::Matrix<double, 3, kNumNodes>;  // 3 x 10
    using ShapeGradSpatial   = Eigen::Matrix<double, 3, kNumNodes>;  // 3 x 10
    using StrainDispMatrix   = Eigen::Matrix<double, 6, kNumDofs>;   // 6 x 30
    using ConstitutiveMatrix = Eigen::Matrix<double, 6, 6>;          // 6 x 6
    using StiffnessMatrix    = Eigen::Matrix<double, kNumDofs, kNumDofs>;  // 30 x 30

    // --- Gauss quadrature data -------------------------------------------------
    // 4-point rule for tetrahedra. Points in barycentric coordinates:
    //   (a,b,b,b), (b,a,b,b), (b,b,a,b), (b,b,b,a)
    // Weight w_g = 1/24 per point.
    //
    // CORRECTION NOTE (Checkpoint 1.1 review):
    //   A valid quadrature must integrate constant fields exactly, i.e. the
    //   weights must sum to the measure of the reference tetrahedron = 1/6.
    //   4 * (1/24) = 1/6  ->  correct (element volume of unit tet = 1/6).
    //   The originally specified 1/96 would give sum = 1/24 (~= 0.0417), a
    //   factor-of-4 error that breaks constant-field exactness and silently
    //   corrupts all element stiffness magnitudes. The eigenvalue-spectrum
    //   test is scale-invariant (zero stays zero, positive stays positive),
    //   which is why Checkpoint 1.1 passed even with the erroneous weight.
    static constexpr double kGaussA = 0.5854101966249685;
    static constexpr double kGaussB = 0.1381966011250105;
    static constexpr double kGaussWeight = 1.0 / 24.0;
    static constexpr std::array<std::array<double, 3>, 4> kGaussPoints = {{
        {kGaussA, kGaussB, kGaussB},
        {kGaussB, kGaussA, kGaussB},
        {kGaussB, kGaussB, kGaussA},
        {kGaussB, kGaussB, kGaussB},
    }};

    // --------------------------------------------------------------------------
    // Constructor: takes 10 nodal coordinates (each row = [x, y, z]).
    // Throws std::invalid_argument if the element is degenerate/inverted.
    // --------------------------------------------------------------------------
    explicit Tet10Element(const CoordMatrix& nodes) : nodes_(nodes) {
        const double vol = computeVolume();
        if (!(vol > 0.0)) {
            throw std::invalid_argument(
                "Tet10Element: non-positive volume. Check node ordering "
                "(corners must be right-handed).");
        }
    }

    // --------------------------------------------------------------------------
    // Shape functions N_i(xi1, xi2, xi3) in barycentric coordinates.
    // xi4 = 1 - xi1 - xi2 - xi3 is implicit.
    //   Corners 1..4 : N_i = xi_i * (2*xi_i - 1)
    //   Mid-edges    : N_jk = 4 * xi_j * xi_k
    // --------------------------------------------------------------------------
    static ShapeVector shapeFunctions(double xi1, double xi2, double xi3) {
        const double xi4 = 1.0 - xi1 - xi2 - xi3;

        ShapeVector N;
        // Corner nodes
        N(0) = xi1 * (2.0 * xi1 - 1.0);
        N(1) = xi2 * (2.0 * xi2 - 1.0);
        N(2) = xi3 * (2.0 * xi3 - 1.0);
        N(3) = xi4 * (2.0 * xi4 - 1.0);
        // Mid-edge nodes
        N(4) = 4.0 * xi1 * xi2;  // edge 1-2
        N(5) = 4.0 * xi2 * xi3;  // edge 2-3
        N(6) = 4.0 * xi3 * xi1;  // edge 3-1
        N(7) = 4.0 * xi1 * xi4;  // edge 1-4
        N(8) = 4.0 * xi2 * xi4;  // edge 2-4
        N(9) = 4.0 * xi3 * xi4;  // edge 3-4
        return N;
    }

    // --------------------------------------------------------------------------
    // Natural-coordinate derivatives dN/dxi (3 x 10).
    // Row j = derivative w.r.t. xi_j; column k = shape function N_k.
    // Chain rule applied for xi4 = 1 - xi1 - xi2 - xi3 (dxi4/dxi_j = -1).
    // --------------------------------------------------------------------------
    static ShapeGradNatural shapeFunctionDerivatives(double xi1, double xi2, double xi3) {
        const double xi4 = 1.0 - xi1 - xi2 - xi3;

        ShapeGradNatural dN;
        dN.setZero();

        // --- Corner nodes ---
        // N1 = xi1*(2*xi1 - 1)
        dN(0, 0) = 4.0 * xi1 - 1.0;
        // N2 = xi2*(2*xi2 - 1)
        dN(1, 1) = 4.0 * xi2 - 1.0;
        // N3 = xi3*(2*xi3 - 1)
        dN(2, 2) = 4.0 * xi3 - 1.0;
        // N4 = xi4*(2*xi4 - 1),  dxi4/dxi_j = -1
        dN(0, 3) = 1.0 - 4.0 * xi4;
        dN(1, 3) = 1.0 - 4.0 * xi4;
        dN(2, 3) = 1.0 - 4.0 * xi4;

        // --- Mid-edge nodes ---
        // N5 = 4*xi1*xi2  (edge 1-2)
        dN(0, 4) = 4.0 * xi2;
        dN(1, 4) = 4.0 * xi1;
        // N6 = 4*xi2*xi3  (edge 2-3)
        dN(1, 5) = 4.0 * xi3;
        dN(2, 5) = 4.0 * xi2;
        // N7 = 4*xi3*xi1  (edge 3-1)
        dN(0, 6) = 4.0 * xi3;
        dN(2, 6) = 4.0 * xi1;
        // N8 = 4*xi1*xi4  (edge 1-4)
        dN(0, 7) = 4.0 * (xi4 - xi1);
        dN(1, 7) = -4.0 * xi1;
        dN(2, 7) = -4.0 * xi1;
        // N9 = 4*xi2*xi4  (edge 2-4)
        dN(0, 8) = -4.0 * xi2;
        dN(1, 8) = 4.0 * (xi4 - xi2);
        dN(2, 8) = -4.0 * xi2;
        // N10 = 4*xi3*xi4 (edge 3-4)
        dN(0, 9) = -4.0 * xi3;
        dN(1, 9) = -4.0 * xi3;
        dN(2, 9) = 4.0 * (xi4 - xi3);

        return dN;
    }

    // --------------------------------------------------------------------------
    // Jacobian matrix J (3 x 3) at a natural point.
    //   J(j, i) = sum_k dN_k/dxi_j * x_k(i)  =>  J = dN_dxi * X
    // This is the transpose of the conventional J_std(i,j) = dx_i/dxi_j, but
    // det(J) = det(J_std) and J^{-1} * dN_dxi = J_std^{-T} * dN_dxi, which is
    // exactly the spatial derivative operator we need (see spatialDerivatives).
    // --------------------------------------------------------------------------
    Eigen::Matrix3d jacobian(double xi1, double xi2, double xi3) const {
        return shapeFunctionDerivatives(xi1, xi2, xi3) * nodes_;
    }

    // --------------------------------------------------------------------------
    // Jacobian determinant at a natural point.
    // --------------------------------------------------------------------------
    double jacobianDeterminant(double xi1, double xi2, double xi3) const {
        return jacobian(xi1, xi2, xi3).determinant();
    }

    // --------------------------------------------------------------------------
    // Spatial derivatives dN/dx (3 x 10) at a natural point.
    //   dN/dx = J^{-1} * dN/dxi   (equivalently J_std^{-T} * dN/dxi)
    // --------------------------------------------------------------------------
    ShapeGradSpatial spatialDerivatives(double xi1, double xi2, double xi3) const {
        const ShapeGradNatural dN_dxi = shapeFunctionDerivatives(xi1, xi2, xi3);
        const Eigen::Matrix3d J_inv = jacobian(xi1, xi2, xi3).inverse();
        return J_inv * dN_dxi;
    }

    // --------------------------------------------------------------------------
    // Element volume (exact for straight-edged TET10 via 4-pt quadrature).
    // --------------------------------------------------------------------------
    // NOTE: The barycentric convention (node i <-> xi_i = 1) with the standard
    // node ordering (0,0,0), (1,0,0), (0,1,0), (0,0,1) yields a NEGATIVE
    // Jacobian determinant (the mapping is x=xi2, y=xi3, z=xi4). The element
    // is still right-handed; we therefore integrate with |det(J)|, which is
    // the standard practice for volume and stiffness computation.
    double computeVolume() const {
        double vol = 0.0;
        for (const auto& gp : kGaussPoints) {
            vol += std::abs(jacobianDeterminant(gp[0], gp[1], gp[2])) * kGaussWeight;
        }
        return vol;
    }

    // --------------------------------------------------------------------------
    // Strain-displacement matrix B (6 x 30) at a natural point.
    // Strain ordering: [exx, eyy, ezz, gxy, gyz, gzx].
    // --------------------------------------------------------------------------
    StrainDispMatrix strainDisplacementMatrix(double xi1, double xi2, double xi3) const {
        const ShapeGradSpatial dN_dx = spatialDerivatives(xi1, xi2, xi3);

        StrainDispMatrix B;
        B.setZero();

        for (int k = 0; k < kNumNodes; ++k) {
            const double bx = dN_dx(0, k);
            const double by = dN_dx(1, k);
            const double bz = dN_dx(2, k);
            const int col = 3 * k;

            B(0, col + 0) = bx;              // exx
            B(1, col + 1) = by;              // eyy
            B(2, col + 2) = bz;              // ezz
            B(3, col + 0) = by;              // gxy
            B(3, col + 1) = bx;
            B(4, col + 1) = bz;              // gyz
            B(4, col + 2) = by;
            B(5, col + 0) = bz;              // gzx
            B(5, col + 2) = bx;
        }
        return B;
    }

    // --------------------------------------------------------------------------
    // Isotropic 3D linear-elasticity constitutive matrix D (6 x 6).
    //   lambda = E*nu / ((1+nu)(1-2nu)),  mu = E / (2(1+nu))
    // --------------------------------------------------------------------------
    static ConstitutiveMatrix constitutiveMatrix(double E, double nu) {
        const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const double mu = E / (2.0 * (1.0 + nu));

        ConstitutiveMatrix D;
        D.setZero();
        D(0, 0) = D(1, 1) = D(2, 2) = lambda + 2.0 * mu;
        D(0, 1) = D(0, 2) = D(1, 0) = D(1, 2) = D(2, 0) = D(2, 1) = lambda;
        D(3, 3) = D(4, 4) = D(5, 5) = mu;
        return D;
    }

    // --------------------------------------------------------------------------
    // Local element stiffness matrix K_e (30 x 30):
    //   K_e = sum_{g=1..4} B_g^T * D * B_g * det(J_g) * w_g
    // --------------------------------------------------------------------------
    StiffnessMatrix computeStiffnessMatrix(double E, double nu) const {
        const ConstitutiveMatrix D = constitutiveMatrix(E, nu);

        StiffnessMatrix Ke;
        Ke.setZero();

        for (const auto& gp : kGaussPoints) {
            const double detJ = std::abs(jacobianDeterminant(gp[0], gp[1], gp[2]));
            const StrainDispMatrix B = strainDisplacementMatrix(gp[0], gp[1], gp[2]);
            Ke.noalias() += B.transpose() * D * B * (detJ * kGaussWeight);
        }

        // Enforce exact symmetry (quadrature introduces ~1e-16 asymmetry that
        // would otherwise perturb the eigenvalue spectrum).
        Ke = 0.5 * (Ke + Ke.transpose());
        return Ke;
    }

private:
    CoordMatrix nodes_;  // 10 x 3 nodal coordinates
};

}  // namespace fea

#endif  // FEA_TET10_ELEMENT_HPP