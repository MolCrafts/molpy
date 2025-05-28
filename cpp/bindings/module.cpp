#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "molpy_core/Frame.hpp"
#include "molpy_core/NestedFrame.hpp"
#include "molpy_core/TensorBlock.hpp"

namespace py = pybind11;

PYBIND11_MODULE(molpy_core, m) {
    m.doc() = "Molpy core C++ backend";

    py::class_<molpy::Frame, std::shared_ptr<molpy::Frame>>(m, "Frame")
        .def(py::init<>())
        .def("add_table", &molpy::Frame::add_table)
        .def("table", &molpy::Frame::table, py::return_value_policy::reference)
        .def_static("concat", &molpy::Frame::concat)
        .def("size", &molpy::Frame::size)
        ;

    py::class_<molpy::NestedFrame, std::shared_ptr<molpy::NestedFrame>>(m, "NestedFrame")
        .def(py::init<>())
        .def("add_table", &molpy::NestedFrame::add_table)
        .def("add_frame", &molpy::NestedFrame::add_frame)
        .def("table", &molpy::NestedFrame::table, py::return_value_policy::reference)
        .def("subframe", &molpy::NestedFrame::subframe, py::return_value_policy::reference)
        .def("slice", &molpy::NestedFrame::slice)
        .def("transpose", &molpy::NestedFrame::transpose)
        .def_static("concat", &molpy::NestedFrame::concat)
        ;

    py::class_<molpy::TensorBlock>(m, "TensorBlock")
        .def(py::init<const xt::xarray<double>&,
                      const std::vector<std::vector<std::string>>&>())
        .def("data", &molpy::TensorBlock::data, py::return_value_policy::reference)
        .def("labels", &molpy::TensorBlock::labels)
        .def("shape", &molpy::TensorBlock::shape)
        .def("transpose", &molpy::TensorBlock::transpose)
        .def("broadcast_to", &molpy::TensorBlock::broadcast_to)
        ;
}
