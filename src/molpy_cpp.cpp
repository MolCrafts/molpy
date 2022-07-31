#include <pybind11/pybind11.h>
#include "randomWalk.h"

namespace py = pybind11;

PYBIND11_MODULE(molpy_cpp, m) {
    m.doc() = "random walk kernel";

    py::class_<molpy::_Modeller>(m, "_Modeller")
        .def(py::init<double, double, double>());

    py::class_<molpy::SimpleRandomWalk, molpy::_Modeller>(m, "SimpleRandomWalk")
        .def(py::init<int, int, int>())
        // .def("find_start", &molpy::SimpleRandomWalk::find_start)
        .def("find_start", &molpy::SimpleRandomWalk::find_start)
        .def("walk_once", static_cast<std::vector<vec3> (molpy::SimpleRandomWalk::*)(int, double)>(&molpy::SimpleRandomWalk::walk_once))
        .def("walk_once", static_cast<std::vector<vec3> (molpy::SimpleRandomWalk::*)(int, double, vec3)>(&molpy::SimpleRandomWalk::walk_once));
        // .def("walk_once", &molpy::SimpleRandomWalk::walk_once);
}