#ifndef ModellerBase_H_
#define ModellerBase_H_

#include <vector>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using vec3 = std::vector<double>;

namespace molpy {

class _Modeller {

    public:

        _Modeller(double lx, double ly, double lz);
        ~_Modeller();

    protected:
        double lx, ly, lz;
        std::default_random_engine rng;
        const double PI = 3.141592653589793;

};

}

#endif // ModellerBase_H_