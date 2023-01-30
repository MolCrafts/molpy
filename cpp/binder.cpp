#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <graph.h>
#include "test.h"

namespace py = pybind11;

template<typename datatype>
void register_graph(py::module &m) {
    using Graph = molpy::Graph<datatype>;
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("has_edge", &Graph::has_edge, "Check if an edge exists")
        .def("set_edge", &Graph::set_edge, "Set an edge")
        .def("has_vertex", &Graph::has_vertex, "Check if a vertex exists")
        .def("set_vertex", &Graph::set_vertex, "Set a vertex")
        .def("get_vertex_label", &Graph::get_vertex_label, "Get the label of a vertex")
        .def("get_num_of_vertices", &Graph::get_num_of_vertices, "Get the number of vertices")
        .def("display", &Graph::display)
        .def("breadth_first_search", &Graph::breadth_first_search)
        .def("depth_first_search", &Graph::depth_first_search);
}


PYBIND11_MODULE(molpy_cpp, m) {
    m.doc() = "MolPy C++ module"; // optional module docstring
    m.def("test_convert_1d", &test_convert_1d, "Test convert 1d array");
    register_graph<int>(m);
}
