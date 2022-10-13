#ifndef MP_SIMPLE_RANDOM_WALK_H
#define MP_SIMPLE_RANDOM_WALK_H

#include <pybind11/pybind11.h> 
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "random_park.h"

namespace py = pybind11;

namespace MOLPY_NS {

class SimpleRandomWalk
{
private:
    /* data */
public:
    SimpleRandomWalk(/* args */);
    ~SimpleRandomWalk();
    
    std::vector<double> find_start(int seed);

    std::vector<double> _walk(int, double, std::vector<double> start_point, int);

    py::array walk(int, double, py::array_t<double, py::array::c_style | py::array::forcecast>, int);

};

} // MOLPY_NS

#endif

