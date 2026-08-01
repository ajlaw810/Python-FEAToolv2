#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>

namespace py = pybind11;

// A simple function: add two integers
int add(int a, int b) {
    return a + b;
}

// A performance-oriented function: compute the nth Fibonacci number iteratively
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

// A function that processes a vector of doubles (e.g., scale each element)
std::vector<double> scale_vector(const std::vector<double>& values, double factor) {
    std::vector<double> result;
    result.reserve(values.size());
    for (double v : values) {
        result.push_back(v * factor);
    }
    return result;
}

// A class example: a simple Counter with state
class Counter {
public:
    Counter(int start = 0) : count_(start) {}

    void increment(int step = 1) { count_ += step; }
    int value() const { return count_; }
    void reset() { count_ = 0; }

private:
    int count_;
};

PYBIND11_MODULE(core, m) {
    m.doc() = "C++ core module bound with pybind11";

    m.def("add", &add, "Add two integers",
          py::arg("a"), py::arg("b"));

    m.def("fibonacci", &fibonacci, "Compute the nth Fibonacci number iteratively",
          py::arg("n"));

    m.def("scale_vector", &scale_vector, "Scale each element of a vector by a factor",
          py::arg("values"), py::arg("factor"));

    py::class_<Counter>(m, "Counter")
        .def(py::init<int>(), py::arg("start") = 0)
        .def("increment", &Counter::increment, py::arg("step") = 1)
        .def("value", &Counter::value)
        .def("reset", &Counter::reset);
}