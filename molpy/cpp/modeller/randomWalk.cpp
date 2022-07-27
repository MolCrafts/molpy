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

std::vector<vec3> molpy::SimpleRandomWalk::walk_once(int nsteps, double stepsize) {

    vec3 next = this->find_start();
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

    // return py::array(py::buffer_info(
    //     traj.data(),  // data as contiguous array
    //     sizeof(double),  // size of one element
    //     py::format_descriptor<double>::format(), // data type   
    //     2,  // ndim
    //     {traj.size(), 3},   // shape: (nsteps, 3)
    //     {sizeof(double)*3, sizeof(double)}  // strides
    // ));
    return traj;

}

