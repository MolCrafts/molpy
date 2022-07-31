#include "randomWalk.h"

molpy::SimpleRandomWalk::SimpleRandomWalk(int lx, int ly, int lz):
    _Modeller(lx, ly, lz), theta(0, 2*PI), phi(0, PI)
{

}

molpy::SimpleRandomWalk::~SimpleRandomWalk() {}

vec3 molpy::SimpleRandomWalk::find_start() {

    std::uniform_real_distribution<double> ex(this->lx);
    std::uniform_real_distribution<double> ey(this->ly);
    std::uniform_real_distribution<double> ez(this->lz);

    vec3 start = {ex(this->rng), ey(this->rng), ez(this->rng)};
    return start;

}

// py::array molpy::SimpleRandomWalk::find_start() {
//     vec3 start = this->_find_start();
//     return py::array(
//         py::buffer_info(
//             start.data(),
//             sizeof(double),
//             py::format_descriptor<double>::format(),
//             1,
//             {3},
//             {sizeof(double)}
//         )
//     );
// }

std::vector<vec3> molpy::SimpleRandomWalk::walk_once(int nsteps, double stepsize, vec3 start) {

    vec3 next = start;
    double r = stepsize;

    std::vector<vec3> traj;
    traj.push_back(next);

    for (int i=0; i<nsteps-1; i++) {

        double theta = this->theta(this->rng);
        double phi = this->phi(this->rng);
        double x = r*cos(theta)*sin(phi);
        double y = r*sin(theta)*sin(phi);
        double z = r*cos(phi);
        next[0] += x;
        next[1] += y;
        next[2] += z;

        traj.push_back(next);

    }
    return traj;

}

std::vector<vec3> molpy::SimpleRandomWalk::walk_once(int nsteps, double stepsize) {
    
    vec3 start = this->find_start();
    return this->walk_once(nsteps, stepsize, start);
    
}

// py::array molpy::SimpleRandomWalk::walk_once(int nsteps, double stepsize) {
//     std::vector<vec3> traj = this->_walk_once(nsteps, stepsize);
//     return py::array(
//         py::buffer_info(
//             traj.data(),
//             sizeof(double),
//             py::format_descriptor<double>::format(),
//             2,
//             {nsteps, 3},
//             {sizeof(double), sizeof(double)*3}
//         )
//     );
// }
