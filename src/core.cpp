#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "global_solver.hpp"
#include "tet10_element.hpp"

namespace nb = nanobind;

// Checkpoint 0 functions & classes
int add(int a, int b) {
    return a + b;
}

uint64_t fibonacci(uint64_t n) {
    if (n <= 1) return n;
    uint64_t a = 0, b = 1;
    for (uint64_t i = 2; i <= n; ++i) {
        uint64_t temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

std::vector<double> scale_vector(const std::vector<double>& values, double factor) {
    std::vector<double> result;
    result.reserve(values.size());
    for (double v : values) {
        result.push_back(v * factor);
    }
    return result;
}

class Counter {
public:
    Counter(int start = 0) : count_(start) {}
    void increment(int step = 1) { count_ += step; }
    int value() const { return count_; }
    void reset() { count_ = 0; }
private:
    int count_;
};

// Checkpoint 2.1 FEASolver
class FEASolver {
public:
    FEASolver(
        nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> nodes,
        nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> elements,
        double E, double nu)
        : E_(E), nu_(nu) {

        if (nodes.ndim() != 2 || nodes.shape(1) != 3) {
            throw std::invalid_argument("nodes must be a 2D array of shape (N_nodes, 3)");
        }
        if (elements.ndim() != 2 || elements.shape(1) != 10) {
            throw std::invalid_argument("elements must be a 2D array of shape (N_elements, 10)");
        }

        n_nodes_ = nodes.shape(0);
        n_elements_ = elements.shape(0);

        nodes_mat_.resize(n_nodes_, 3);
        if (nodes.dtype() == nb::dtype<double>()) {
            const double* ptr = static_cast<const double*>(nodes.data());
            for (size_t i = 0; i < n_nodes_; ++i) {
                nodes_mat_(i, 0) = ptr[i * 3 + 0];
                nodes_mat_(i, 1) = ptr[i * 3 + 1];
                nodes_mat_(i, 2) = ptr[i * 3 + 2];
            }
        } else if (nodes.dtype() == nb::dtype<float>()) {
            const float* ptr = static_cast<const float*>(nodes.data());
            for (size_t i = 0; i < n_nodes_; ++i) {
                nodes_mat_(i, 0) = static_cast<double>(ptr[i * 3 + 0]);
                nodes_mat_(i, 1) = static_cast<double>(ptr[i * 3 + 1]);
                nodes_mat_(i, 2) = static_cast<double>(ptr[i * 3 + 2]);
            }
        } else {
            throw std::invalid_argument("nodes array must have float64 or float32 dtype");
        }

        elements_0based_.resize(n_elements_, 10);
        conn_1based_.resize(n_elements_, 10);

        if (elements.dtype() == nb::dtype<int32_t>()) {
            const int32_t* ptr = static_cast<const int32_t*>(elements.data());
            for (size_t e = 0; e < n_elements_; ++e) {
                for (size_t a = 0; a < 10; ++a) {
                    int node_id = static_cast<int>(ptr[e * 10 + a]);
                    elements_0based_(e, a) = node_id;
                    conn_1based_(e, a) = node_id + 1;
                }
            }
        } else if (elements.dtype() == nb::dtype<int64_t>()) {
            const int64_t* ptr = static_cast<const int64_t*>(elements.data());
            for (size_t e = 0; e < n_elements_; ++e) {
                for (size_t a = 0; a < 10; ++a) {
                    int node_id = static_cast<int>(ptr[e * 10 + a]);
                    elements_0based_(e, a) = node_id;
                    conn_1based_(e, a) = node_id + 1;
                }
            }
        } else {
            throw std::invalid_argument("elements array must have int32 or int64 dtype");
        }
    }

    void apply_fixed_bc(nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> fixed_node_ids) {
        if (fixed_node_ids.ndim() != 1) {
            throw std::invalid_argument("fixed_node_ids must be a 1D array");
        }
        size_t n_fixed = fixed_node_ids.shape(0);
        fixed_ids_.resize(n_fixed);

        if (fixed_node_ids.dtype() == nb::dtype<int32_t>()) {
            const int32_t* ptr = static_cast<const int32_t*>(fixed_node_ids.data());
            for (size_t i = 0; i < n_fixed; ++i) fixed_ids_(i) = static_cast<int>(ptr[i]);
        } else if (fixed_node_ids.dtype() == nb::dtype<int64_t>()) {
            const int64_t* ptr = static_cast<const int64_t*>(fixed_node_ids.data());
            for (size_t i = 0; i < n_fixed; ++i) fixed_ids_(i) = static_cast<int>(ptr[i]);
        } else {
            throw std::invalid_argument("fixed_node_ids must have int32 or int64 dtype");
        }
    }

    void apply_point_loads(
        nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> load_node_ids,
        nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> forces) {
        if (load_node_ids.ndim() != 1) {
            throw std::invalid_argument("load_node_ids must be a 1D array");
        }
        if (forces.ndim() != 2 || forces.shape(1) != 3) {
            throw std::invalid_argument("forces must be a 2D array of shape (N_loads, 3)");
        }
        size_t n_loads = load_node_ids.shape(0);
        if (forces.shape(0) != n_loads) {
            throw std::invalid_argument("load_node_ids size must match forces rows");
        }

        load_ids_.resize(n_loads);
        if (load_node_ids.dtype() == nb::dtype<int32_t>()) {
            const int32_t* ptr = static_cast<const int32_t*>(load_node_ids.data());
            for (size_t i = 0; i < n_loads; ++i) load_ids_(i) = static_cast<int>(ptr[i]);
        } else if (load_node_ids.dtype() == nb::dtype<int64_t>()) {
            const int64_t* ptr = static_cast<const int64_t*>(load_node_ids.data());
            for (size_t i = 0; i < n_loads; ++i) load_ids_(i) = static_cast<int>(ptr[i]);
        } else {
            throw std::invalid_argument("load_node_ids must have int32 or int64 dtype");
        }

        forces_mat_.resize(n_loads, 3);
        if (forces.dtype() == nb::dtype<double>()) {
            const double* ptr = static_cast<const double*>(forces.data());
            for (size_t i = 0; i < n_loads; ++i) {
                forces_mat_(i, 0) = ptr[i * 3 + 0];
                forces_mat_(i, 1) = ptr[i * 3 + 1];
                forces_mat_(i, 2) = ptr[i * 3 + 2];
            }
        } else if (forces.dtype() == nb::dtype<float>()) {
            const float* ptr = static_cast<const float*>(forces.data());
            for (size_t i = 0; i < n_loads; ++i) {
                forces_mat_(i, 0) = static_cast<double>(ptr[i * 3 + 0]);
                forces_mat_(i, 1) = static_cast<double>(ptr[i * 3 + 1]);
                forces_mat_(i, 2) = static_cast<double>(ptr[i * 3 + 2]);
            }
        } else {
            throw std::invalid_argument("forces array must have float64 or float32 dtype");
        }
    }

    void solve() {
        solver_ = std::make_unique<fea::GlobalSolver>(nodes_mat_, conn_1based_, E_, nu_);
        if (fixed_ids_.size() > 0) {
            solver_->applyFixedConstraints(fixed_ids_);
        }
        if (load_ids_.size() > 0) {
            solver_->applyPointLoads(load_ids_, forces_mat_);
        }
        solver_->solve();
        solved_ = true;
    }

    nb::ndarray<nb::numpy, double, nb::shape<-1, 3>> get_displacements() {
        if (!solved_) {
            throw std::runtime_error("Must call solve() before getting displacements");
        }
        const Eigen::VectorXd& u = solver_->u();
        double* disp_data = new double[n_nodes_ * 3];
        for (size_t i = 0; i < n_nodes_; ++i) {
            disp_data[i * 3 + 0] = u(3 * i + 0);
            disp_data[i * 3 + 1] = u(3 * i + 1);
            disp_data[i * 3 + 2] = u(3 * i + 2);
        }
        size_t shape[2] = { n_nodes_, 3 };
        nb::capsule owner(disp_data, [](void *p) noexcept { delete[] static_cast<double*>(p); });
        return nb::ndarray<nb::numpy, double, nb::shape<-1, 3>>(disp_data, 2, shape, owner);
    }

    nb::ndarray<nb::numpy, double, nb::shape<-1>> get_von_mises_stresses() {
        if (!solved_) {
            throw std::runtime_error("Must call solve() before getting von Mises stresses");
        }
        const Eigen::VectorXd& u_global = solver_->u();

        double a = fea::Tet10Element::kGaussA;
        double b = fea::Tet10Element::kGaussB;

        Eigen::Matrix4d M_GP;
        M_GP << a, b, b, b,
                b, a, b, b,
                b, b, a, b,
                b, b, b, a;
        Eigen::Matrix4d M_GP_inv = M_GP.inverse();

        Eigen::Matrix<double, 10, 4> N_nodes_lin;
        N_nodes_lin <<
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            0.5, 0.5, 0.0, 0.0,
            0.0, 0.5, 0.5, 0.0,
            0.5, 0.0, 0.5, 0.0,
            0.5, 0.0, 0.0, 0.5,
            0.0, 0.5, 0.0, 0.5,
            0.0, 0.0, 0.5, 0.5;

        Eigen::Matrix<double, 10, 4> E_extrap = N_nodes_lin * M_GP_inv;
        fea::Tet10Element::ConstitutiveMatrix D = fea::Tet10Element::constitutiveMatrix(E_, nu_);

        Eigen::MatrixXd sigma_sum = Eigen::MatrixXd::Zero(n_nodes_, 6);
        Eigen::VectorXi count = Eigen::VectorXi::Zero(n_nodes_);

        for (size_t e = 0; e < n_elements_; ++e) {
            fea::Tet10Element::CoordMatrix elem_nodes;
            Eigen::Matrix<double, 30, 1> u_elem;

            for (int i = 0; i < 10; ++i) {
                int node_id = elements_0based_(e, i);
                elem_nodes.row(i) = nodes_mat_.row(node_id);
                u_elem(3 * i + 0) = u_global(3 * node_id + 0);
                u_elem(3 * i + 1) = u_global(3 * node_id + 1);
                u_elem(3 * i + 2) = u_global(3 * node_id + 2);
            }

            fea::Tet10Element elem(elem_nodes);

            Eigen::Matrix<double, 4, 6> sigma_GP;
            for (int g = 0; g < 4; ++g) {
                const auto& gp = fea::Tet10Element::kGaussPoints[g];
                fea::Tet10Element::StrainDispMatrix B = elem.strainDisplacementMatrix(gp[0], gp[1], gp[2]);
                Eigen::Matrix<double, 6, 1> strain = B * u_elem;
                Eigen::Matrix<double, 6, 1> stress = D * strain;
                sigma_GP.row(g) = stress.transpose();
            }

            Eigen::Matrix<double, 10, 6> sigma_elem = E_extrap * sigma_GP;

            for (int i = 0; i < 10; ++i) {
                int node_id = elements_0based_(e, i);
                sigma_sum.row(node_id) += sigma_elem.row(i);
                count(node_id) += 1;
            }
        }

        double* vm_data = new double[n_nodes_];
        for (size_t i = 0; i < n_nodes_; ++i) {
            if (count(i) > 0) {
                Eigen::Matrix<double, 1, 6> sig_avg = sigma_sum.row(i) / static_cast<double>(count(i));
                double sxx = sig_avg(0);
                double syy = sig_avg(1);
                double szz = sig_avg(2);
                double sxy = sig_avg(3);
                double syz = sig_avg(4);
                double szx = sig_avg(5);

                double vm = std::sqrt(0.5 * ( (sxx - syy)*(sxx - syy) + (syy - szz)*(syy - szz) + (szz - sxx)*(szz - sxx)
                                             + 6.0 * (sxy*sxy + syz*syz + szx*szx) ));
                vm_data[i] = vm;
            } else {
                vm_data[i] = 0.0;
            }
        }

        size_t shape[1] = { n_nodes_ };
        nb::capsule owner(vm_data, [](void *p) noexcept { delete[] static_cast<double*>(p); });
        return nb::ndarray<nb::numpy, double, nb::shape<-1>>(vm_data, 1, shape, owner);
    }

private:
    size_t n_nodes_;
    size_t n_elements_;
    double E_;
    double nu_;
    Eigen::MatrixXd nodes_mat_;
    Eigen::MatrixXi elements_0based_;
    Eigen::MatrixXi conn_1based_;
    Eigen::VectorXi fixed_ids_;
    Eigen::VectorXi load_ids_;
    Eigen::MatrixXd forces_mat_;
    std::unique_ptr<fea::GlobalSolver> solver_;
    bool solved_ = false;
};

NB_MODULE(core, m) {
    m.doc() = "C++ core module bound with nanobind";

    m.def("add", &add, "Add two integers", nb::arg("a"), nb::arg("b"));
    m.def("fibonacci", &fibonacci, "Compute the nth Fibonacci number iteratively", nb::arg("n"));
    m.def("scale_vector", &scale_vector, "Scale each element of a vector by a factor", nb::arg("values"), nb::arg("factor"));

    nb::class_<Counter>(m, "Counter")
        .def(nb::init<int>(), nb::arg("start") = 0)
        .def("increment", &Counter::increment, nb::arg("step") = 1)
        .def("value", &Counter::value)
        .def("reset", &Counter::reset);

    nb::class_<FEASolver>(m, "FEASolver")
        .def(nb::init<nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>,
                      nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>,
                      double, double>(),
             nb::arg("nodes"), nb::arg("elements"), nb::arg("E"), nb::arg("nu"))
        .def("apply_fixed_bc", &FEASolver::apply_fixed_bc, nb::arg("fixed_node_ids"))
        .def("apply_point_loads", &FEASolver::apply_point_loads, nb::arg("load_node_ids"), nb::arg("forces"))
        .def("solve", &FEASolver::solve)
        .def("get_displacements", &FEASolver::get_displacements)
        .def("get_von_mises_stresses", &FEASolver::get_von_mises_stresses);
}