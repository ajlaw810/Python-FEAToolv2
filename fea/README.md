# TET10 Element — Checkpoint 1.1

Self-contained C++20 implementation of a **10-node Quadratic Tetrahedron (TET10)** element stiffness matrix assembly, with an eigenvalue-spectrum validation test (Checkpoint 1.1).

## Files

| File | Purpose |
|------|---------|
| `tet10_element.hpp` | Header-only `fea::Tet10Element`: shape functions, Jacobian, B & D matrices, 4-point Gauss quadrature, 30x30 stiffness assembly |
| `main_tet10_checkpoint.cpp` | Standalone `main()`: unit TET10, K_e computation, 30-eigenvalue spectrum test with automated assertions |
| `CMakeLists.txt` | Optional CMake build |

## Mathematical Formulation

- **Shape functions** (barycentric coords, xi4 = 1 - xi1 - xi2 - xi3):
  - Corners 1-4: `N_i = xi_i * (2*xi_i - 1)`
  - Mid-edges 5-10: `N5 = 4*xi1*xi2`, `N6 = 4*xi2*xi3`, `N7 = 4*xi3*xi1`, `N8 = 4*xi1*xi4`, `N9 = 4*xi2*xi4`, `N10 = 4*xi3*xi4`
- **Spatial derivatives**: `dN/dx = J^-1 * dN/dxi` via the 3x3 inverse Jacobian
- **Integration**: 4-point Gauss rule, points `(a,b,b,b)`, `(b,a,b,b)`, `(b,b,a,b)`, `(b,b,b,a)` with `a = 0.5854101966249685`, `b = 0.1381966011250105`, weight `w_g = 1/24` per point (4 points sum to `1/6`, the reference tet volume — required for constant-field exactness)
- **Stiffness**: `K_e = sum_g B^T D B det(J) w_g` (30x30), isotropic `D` from `E`, `nu`

## Checkpoint 1.1 Validation

Builds a **unit tetrahedron** (corners `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)`, mid-edge nodes at exact midpoints) with **E = 210 GPa, nu = 0.3**, then:

1. Computes all 30 eigenvalues via `Eigen::SelfAdjointEigenSolver`
2. Sorts and prints them
3. Asserts **exactly 6** eigenvalues are effectively zero (`< 1e-3 * lambda_max`) -> 6 rigid body modes
4. Asserts the remaining **24** eigenvalues are strictly positive -> positive definiteness under constraint
5. Prints `[CHECKPOINT 1.1 PASSED]: Rigid Body Modes and Positive Definiteness Verified!`

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
g++ -std=c++20 -O2 -I<path-to-eigen> main_tet10_checkpoint.cpp -o tet10_checkpoint
./tet10_checkpoint
```

Example (Eigen extracted to `C:\eigen`):
```bash
g++ -std=c++20 -O2 -IC:\eigen main_tet10_checkpoint.cpp -o tet10_checkpoint.exe
.\tet10_checkpoint.exe
```

### CMake

```bash
cd fea
cmake -B build -DEIGEN3_INCLUDE_DIR=<path-to-eigen>
cmake --build build --config Release
./build/tet10_checkpoint            # Linux/macOS
.\build\Release\tet10_checkpoint.exe  # Windows/MSVC
```

If Eigen is installed system-wide, `find_package(Eigen3)` locates it automatically and `-DEIGEN3_INCLUDE_DIR` is unnecessary.

### MSVC (cl.exe) directly

```bat
cl /std:c++20 /O2 /EHsc /I<path-to-eigen> main_tet10_checkpoint.cpp
main_tet10_checkpoint.exe
```

## Expected Output

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

The program exits with code `0` on success and non-zero on failure.