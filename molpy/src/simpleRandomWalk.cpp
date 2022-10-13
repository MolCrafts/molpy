#include "simpleRandomWalk.h"
#include <iostream>

MOLPY_NS::SimpleRandomWalk::SimpleRandomWalk(/* args */)
{
}

MOLPY_NS::SimpleRandomWalk::~SimpleRandomWalk()
{
}

std::vector<double> MOLPY_NS::SimpleRandomWalk::find_start(int seed)
{
    RanPark random = RanPark(seed);
    std::vector<double> start_point = {random.uniform(), random.uniform(), random.uniform()};
    return start_point;
}

std::vector<double> MOLPY_NS::SimpleRandomWalk::_walk(int nsteps, double step_size, std::vector<double> start_point, int seed) {
    RanPark random = RanPark(seed);
    std::vector<double> point = start_point;
    std::vector<double> walk;
    for (int i = 0; i < nsteps; i++) {
        walk.push_back(point[0]);
        walk.push_back(point[1]);
        walk.push_back(point[2]);

        double theta = random.uniform() * 2 * M_PI;
        double phi = random.uniform() * M_PI;
        double x = step_size * sin(phi) * cos(theta);
        double y = step_size * sin(phi) * sin(theta);
        double z = step_size * cos(phi);

        point[0] += x;
        point[1] += y;
        point[2] += z;
    }
    return walk;
}

py::array MOLPY_NS::SimpleRandomWalk::walk(int nsteps, double step_size, py::array_t<double, py::array::c_style | py::array::forcecast> start_point, int seed) {
    std::vector<double> start(start_point.size());
    std::memcpy(start.data(), start_point.data(), start_point.size() * sizeof(double));
    std::vector<double> walk = _walk(nsteps, step_size, start, seed);

    auto result = py::array_t<double>(nsteps*3);
    auto result_buffer = result.request();
    double *result_ptr = (double *) result_buffer.ptr;

    std::memcpy(result_ptr, walk.data(), walk.size() * sizeof(double));

    return result.reshape({nsteps, 3});

}

PYBIND11_MODULE(molpy_kernel, m) {
    py::class_<MOLPY_NS::SimpleRandomWalk>(m, "SimpleRandomWalk")
        .def(py::init<>())
        .def("walk", &MOLPY_NS::SimpleRandomWalk::walk);
}