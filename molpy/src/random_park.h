#ifndef MP_RANPARK_H
#define MP_RANPARK_H

namespace MOLPY_NS {

class RanPark {
    public:
        RanPark(int);
        double uniform();
        double gaussian();
        void reset(int);
        void reset(int, double *);
        int state();

    private:
        int seed, save;
        double second;
};

}

#endif