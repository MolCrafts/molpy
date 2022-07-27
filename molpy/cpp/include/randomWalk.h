#ifndef RandomWalk_h_
#define RandomWalk_h_

#include "modellerBase.h"

namespace molpy {


class SimpleRandomWalk: public _Modeller {

    public:

        SimpleRandomWalk(int lx, int ly, int lz);
        ~SimpleRandomWalk();
        std::vector<vec3> walk_once(int length, double stepsize);
        vec3 find_start();

    private:

        std::uniform_real_distribution<double> theta;
        std::uniform_real_distribution<double> phi;

};

}

#endif // RandomWalk_h_