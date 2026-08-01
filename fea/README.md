# TET10 Element — Checkpoints 1.1 & 1.2

Self-contained C++20 3D structural FEA engine using **10-node Quadratic Tetrahedron (TET10)** elements, with:

- **Checkpoint 1.1** — element stiffness matrix + eigenvalue-spectrum validation (rigid body modes, positive definiteness)
- **Checkpoint 1.2** — global sparse matrix assembly, boundary conditions, `SimplicialLDLT` solve, and a cantilever beam theoretical benchmark (Euler-Bernoulli tip deflection)

## Files

| File | Purpose |
|------|---------|
| `tet10_element.hpp` | Header-only `fea::Tet10Element`: shape functions, Jacobian, B & D matrices, 4-point Gauss quadrature, 30x30 stiffness assembly |
| `global_solver.hpp` | Header-only `fea::GlobalSolver`: global sparse assembly via `Eigen::Triplet`/`SparseMatrix`, fixed-DOF BCs, point loads, `Eigen::SimplicialLDLT` solve |
| `main_tet10_checkpoint.cpp` | Standalone `main()`: unit TET10, K_e computation, 30-eigenvalue spectrum test with automated assertions |
| `main_beam_checkpoint.cpp` | Standalone `main()`: structured hex mesh → 6-TET10-per-hex (Kuhn split) cantilever beam, global solve, deflection vs. beam theory |
| `CMakeLists.txt` | Optional CMake build (both executables) |

## Mathematical Formulation

- **Shape functions** (barycentric coords, xi4 = 1 - xi1 - xi2 - xi3):
  - Corners 1-4: `N_i = xi_i * (2*xi_i - 1)`
  - Mid-edges 5-10: `N5 = 4*xi1*xi2`, `N6 = 4*xi2*xi3`, `N7 = 4*xi3*xi1`, `N8 = 4*xi1*xi4`, `N9 = 4*xi2*xi4`, `N10 = 4*xi3*xi4`
- **Spatial derivatives**: `dN/dx = J^-1 * dN/dxi` via the 3x3 inverse Jacobian
- **Integration**: 4-point Gauss rule, points `(a,b,b,b)`, `(b,a,b,b)`, `(b,b,a,b)`, `(b,b,b,a)` with `a = 0.5854101966249685`, `b = 0.1381966011250105`, weight `w_g = 1/24` per point (4 points sum to `1/6`, the reference tet volume — required for constant-field exactness)
- **Element stiffness**: `K_e = sum_g B^T D B |det(J)| w_g` (30x30), isotropic `D` from `E`, `nu`
- **Global assembly**: `K = sum_e (3x3 block-scatter of K_e into global DOFs)`, `(3*N_nodes) x (3*N_nodes)` sparse via triplets
- **Boundary conditions**: for each fixed DOF, zero row/col, `K_ii = 1.0`, `F_i = 0.0` (symmetry preserved)
- **Solve**: `K * u = F` via `Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>`

## Checkpoint 1.1 Validation

Builds a **unit tetrahedron** (corners `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)`, mid-edge nodes at exact midpoints) with **E = 210 GPa, nu = 0.3**, then:

1. Computes all 30 eigenvalues via `Eigen::SelfAdjointEigenSolver`
2. Sorts and prints them
3. Asserts **exactly 6** eigenvalues are effectively zero (`< 1e-3 * lambda_max`) -> 6 rigid body modes
4. Asserts the remaining **24** eigenvalues are strictly positive -> positive definiteness under constraint
5. Prints `[CHECKPOINT 1.1 PASSED]: Rigid Body Modes and Positive Definiteness Verified!`

## Checkpoint 1.2 Validation

Builds a **cantilever beam** 10 x 1 x 1 m clamped at `X = 0` with a total downward tip load `F_z = -100,000 N` distributed evenly over all nodes on the `X = L` face:

1. **Structured mesh**: `20 x 2 x 2 = 80` sub-cubes, each split into **6 TET10 elements** via the Freudenthal/Kuhn triangulation around the body diagonal (480 TET10 elements total; mid-edge nodes are created through a sorted-pair edge map so shared edges/faces get unique global midpoint nodes at exact edge midpoints)
2. **Global assembly**: `fea::GlobalSolver` builds `K` (`3*N_nodes x 3*N_nodes` sparse) from `Eigen::Triplet<double>` triplets
3. **BCs**: clamp all nodes on `X = 0` (zero rows/cols, `K_ii = 1`, `F_i = 0`)
4. **Loads**: `F_z = -100000/N_tip` applied to each tip-face node
5. **Solve**: `Eigen::SimplicialLDLT`, asserts `solver.info() == Eigen::Success`
6. **Theory**: `I = w*h^3/12 = 1/12 m^4`, `delta_exact = F*L^3/(3*E*I) = 0.00190476 m` (1.9048 mm)
7. **Assert**: `delta_FEA` = max `|u_z|` at tip -> relative error `|delta_FEA - delta_exact|/delta_exact < 3%`
8. Prints `[CHECKPOINT 1.2 PASSED]: Cantilever Beam Deflection Verified against Beam Theory!`

> **Note on beam theory vs. FEA**: the Euler-Bernoulli formula `FL^3/(3EI)` neglects transverse shear deformation. With `L/h = 10` the Timoshenko shear contribution is only ~0.4%, so the 3% tolerance is comfortably met by the quadratic TET10 discretization.

## Prerequisites

- C++20 compiler (GCC >= 10, Clang >= 12, MSVC >= 2019)
- **Eigen3** (header-only, no compilation needed)

### Getting Eigen

**Option A - download (simplest):** Download from https://eigen.tuxfamily.org and extract, e.g. `eigen-3.4.0.zip` -> `C:\eigen` (contains `C:\eigen\Eigen\Dense`).

**Option B - vcpkg (Windows):**
```bash
vcpkg install eigen3
# then: -DCMAKE_TOOLCHAIN_FILE=<vcpkg>\scripts\buildsystems\vcpkg.cmake
```

**Option C - package manager (Linux/macOS):**
```bash
sudo apt install libeigen3-dev        # Debian/Ubuntu
brew install eigen                    # macOS
```

## Building

### Direct compile with g++ (MinGW-w64 on Windows, or GCC/Clang)

```bash
cd fea
# Checkpoint 1.1
g++ -std=c++20 -O2 -I<path-to-eigen> main_tet10_checkpoint.cpp -o tet10_checkpoint
./tet10_checkpoint

# Checkpoint 1.2
g++ -std=c++20 -O2 -I<path-to-eigen> main_beam_checkpoint.cpp -o beam_checkpoint
./beam_checkpoint
```

Example (Eigen extracted to `C:\eigen`):
```bash
g++ -std=c++20 -O2 -IC:\eigen main_tet10_checkpoint.cpp -o tet10_checkpoint.exe
g++ -std=c++20 -O2 -IC:\eigen main_beam_checkpoint.cpp -o beam_checkpoint.exe
.\tet10_checkpoint.exe
.\beam_checkpoint.exe
```

### CMake

```bash
cd fea
cmake -B build -DEIGEN3_INCLUDE_DIR=<path-to-eigen>
cmake --build build --config Release
./build/tet10_checkpoint            # Linux/macOS
.\build\Release\tet10_checkpoint.exe  # Windows/MSVC
./build/beam_checkpoint
.\build\Release\beam_checkpoint.exe
```

If Eigen is installed system-wide, `find_package(Eigen3)` locates it automatically and `-DEIGEN3_INCLUDE_DIR` is unnecessary.

### MSVC (cl.exe) directly

```bat
cl /std:c++20 /O2 /EHsc /I<path-to-eigen> main_tet10_checkpoint.cpp
cl /std:c++20 /O2 /EHsc /I<path-to-eigen> main_beam_checkpoint.cpp
```

## Expected Output

### Checkpoint 1.1

```
============================================================
 TET10 Checkpoint 1.1 - Eigenvalue Spectrum Test
============================================================
 Material      : E = 2.1e+11 Pa, nu = 0.3
 Element volume: 0.166666666667 (exact = 1/6)
 Stiffness K_e : 30 x 30
 Max |K_ij|    : 1.000000e+11
...
  effectively-zero evals= 6 (expected 6: 3 translations + 3 rotations)
  strictly-positive evals= 24 (expected 24)

[CHECKPOINT 1.1 PASSED]: Rigid Body Modes and Positive Definiteness Verified!
```

### Checkpoint 1.2

```
============================================================
 TET10 Checkpoint 1.2 - Cantilever Beam Benchmark
============================================================
 Geometry     : L = 10 m, w = 1 m, h = 1 m
 Material     : E = 2.1e+11 Pa, nu = 0.3
 Mesh         : 20 x 2 x 2 sub-cubes, 6 TET10 per hex (Kuhn split)
 Nodes        : 1025
 TET10 elems  : 480
 Fixed nodes  : 25 (all DOFs on X = 0)
 Tip nodes    : 25 (F_z distributed on X = L)

 Assembled global system:
 Global system      : 3075 x 3075 DOFs
 Non-zeros in K     : 196167
 Constrained DOFs   : 75

 Solve        : SimplicialLDLT converged (Eigen::Success)

============================================================
 Checkpoint 1.2 Results
============================================================
  I             = 0.08333333 m^4  (w*h^3/12)
  delta_exact   = 1.90476190e-03 m  (F*L^3/(3*E*I))
  delta_FEA     = 1.89874898e-03 m  (max |u_z| at tip)
  min u_z(tip)  = -1.89874898e-03 m  (downward = negative)
  relative error= 0.3157%  (tolerance 3.0000%)

[CHECKPOINT 1.2 PASSED]: Cantilever Beam Deflection Verified against Beam Theory!
```

Both programs exit with code `0` on success and non-zero on failure.