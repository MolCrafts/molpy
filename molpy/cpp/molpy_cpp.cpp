#include <pybind11/pybind11.h>
#include "randomWalk.h"

namespace py = pybind11;


PYBIND11_MODULE(molpy_cpp, m) {
    m.doc() = "random walk kernel";
    py::class_<molpy::SimpleRandomWalk, molpy::_Modeller>(m, "SimpleRandomWalk")
        .def(py::init<int, int, int>())
        .def("find_start", &molpy::SimpleRandomWalk::find_start)
        .def("walk_once", &molpy::SimpleRandomWalk::walk_once);

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

}