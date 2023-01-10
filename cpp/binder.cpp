#include <pybind11/pybind11.h>
#include <graph.h>

namespace py = pybind11;

template<typename datatype>
void register_graph(py::module &m) {
    using Graph = molpy::Graph<datatype>;
    using vector = std::vector<datatype>;
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def(py::init<vector &>())
        .def("set_edge", &Graph::set_edge)
        .def("display", &Graph::display)
        .def("breadth_first_search", &Graph::breadth_first_search)
        .def("depth_first_search", &Graph::depth_first_search);
}


PYBIND11_MODULE(molpy_cpp, m) {
    m.doc() = "MolPy C++ module"; // optional module docstring
    register_graph<int>(m);
}
