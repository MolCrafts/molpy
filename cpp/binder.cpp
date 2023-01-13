#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <graph.h>
// #include <test.h>

namespace py = pybind11;


// // wrap C++ function with NumPy array IO
// py::array_t<int> py_multiply(py::array_t<double, py::array::c_style | py::array::forcecast> array)
// {
//   // allocate std::vector (to pass to the C++ function)
//   std::vector<double> array_vec(array.size());

//   // copy py::array -> std::vector
//   std::memcpy(array_vec.data(),array.data(),array.size()*sizeof(double));

//   // call pure C++ function
//   std::vector<int> result_vec = multiply(array_vec);

//   // allocate py::array (to pass the result of the C++ function to Python)
//   auto result        = py::array_t<int>(array.size());
//   auto result_buffer = result.request();
//   int *result_ptr    = (int *) result_buffer.ptr;

//   // copy std::vector -> py::array
//   std::memcpy(result_ptr,result_vec.data(),result_vec.size()*sizeof(int));

//   return result;
// }

// // wrap as Python module
// PYBIND11_MODULE(molpy_cpp,m)
// {
//   m.doc() = "pybind11 example plugin";

//   m.def("multiply", &py_multiply, "Convert all entries of an 1-D NumPy-array to int and multiply by 10");
// }


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
    register_graph<int>(m);
}
