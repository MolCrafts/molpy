#ifndef ModellerBase_H_
#define ModellerBase_H_

#include <vector>
#include <random>

using vec3 = std::vector<double>;

namespace molpy {

class _Modeller {

    public:

        _Modeller(double lx, double ly, double lz);
        ~_Modeller();
        vec3 find_start();
        vec3 walk_once(int length);

    protected:
        double lx, ly, lz;
        std::default_random_engine rng;
        const double PI = 3.141592653589793;

};

}

#endif // ModellerBase_H_