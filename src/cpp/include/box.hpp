// Copyright (c) 2010-2025 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef BOX_H
#define BOX_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <math.h> // NOLINT(modernize-deprecated-headers): Use std::numbers when c++20 is default.
#include <stdexcept>
#include <vector>

#include "types.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xio.hpp>

/*! \file Box.h
    \brief Represents simulation boxes and contains helpful wrapping functions.
*/

namespace molcpp { namespace constants {
// Constant 2*pi for convenient use everywhere.
constexpr float TWO_PI = 2.0 * M_PI;
}; }; // end namespace molcpp::constants

namespace molcpp { namespace box {

//! Stores box dimensions and provides common routines for wrapping vectors back into the box
/*! Box stores a standard HOOMD simulation box that goes from -L/2 to L/2 in each dimension, allowing Lx, Ly,
 Lz, and triclinic tilt factors xy, xz, and yz to be specified independently.
 *

    A number of utility functions are provided to work with coordinates in boxes. These are provided as
 inlined methods in the header file so they can be called in inner loops without sacrificing performance.
     - wrap()
     - unwrap()

    A Box can represent either a two or three dimensional box. By default, a Box is 3D, but can be set as 2D
 with the method set2D(), or via an optional boolean argument to the constructor. is2D() queries if a Box is
 2D or not. 2D boxes have a "volume" of Lx * Ly, and Lz is set to 0. To keep programming simple, all inputs
 and outputs are still 3-component vectors even for 2D boxes. The third component ignored (assumed set to 0).
*/
class Box
{
public:
    //! Nullary constructor for Cython
    Box()
    {
        m_2d = false; // Assign before calling setL!
        setL(0, 0, 0);
        m_periodic = vec3b{true, true, true};
        m_xy = m_xz = m_yz = 0;
    }

    //! Construct a square/cubic box
    explicit Box(float L, bool _2d = false)
    {
        m_2d = _2d; // Assign before calling setL!
        setL(L, L, L);
        m_periodic = vec3b{true, true, true};
        m_xy = m_xz = m_yz = 0;
    }

    //! Construct an orthorhombic box
    explicit Box(float Lx, float Ly, float Lz, bool _2d = false)
    {
        m_2d = _2d; // Assign before calling setL!
        setL(Lx, Ly, Lz);
        m_periodic = vec3b{true, true, true};
        m_xy = m_xz = m_yz = 0;
    }

    //! Construct a triclinic box
    explicit Box(float Lx, float Ly, float Lz, float xy, float xz, float yz, bool _2d = false)
    {
        m_2d = _2d; // Assign before calling setL!
        setL(Lx, Ly, Lz);
        m_periodic = vec3b{true, true, true};
        m_xy = xy;
        m_xz = xz;
        m_yz = yz;
    }

    bool operator==(const Box& b) const
    {
        return ((this->getL() == b.getL()) && (this->getTiltFactorXY() == b.getTiltFactorXY())
                && (this->getTiltFactorXZ() == b.getTiltFactorXZ())
                && (this->getTiltFactorYZ() == b.getTiltFactorYZ())
                && (this->getPeriodicX() == b.getPeriodicX()) && (this->getPeriodicY() == b.getPeriodicY())
                && (this->getPeriodicZ() == b.getPeriodicZ()));
    }

    bool operator!=(const Box& b) const
    {
        return !(*this == b);
    }

    //! Set L, box lengths, inverses.  Box is also centered at zero.
    void setL(const float Lx, const float Ly, const float Lz)
    {
        if (m_2d)
        {
            m_L = vec3f {Lx, Ly, 0};
            m_Linv = vec3f{1 / m_L[0], 1 / m_L[1], 0};
        }
        else
        {
            m_L = vec3f {Lx, Ly, Lz};
            m_Linv = vec3f{1 / m_L[0], 1 / m_L[1], 1 / m_L[2]};
        }

        m_hi = m_L / 2.0;
        m_lo = -m_hi;
    }

    //! Set whether box is 2D
    void set2D(bool _2d)
    {
        m_2d = _2d;
        m_L[2] = 0;
        m_Linv[2] = 0;
    }

    //! Returns whether box is two dimensional
    bool is2D() const
    {
        return m_2d;
    }

    //! Get the value of Lx
    float getLx() const
    {
        return m_L[0];
    }

    //! Get the value of Ly
    float getLy() const
    {
        return m_L[1];
    }

    //! Get the value of Lz
    float getLz() const
    {
        return m_L[2];
    }

    //! Get current L
    vec3f getL() const
    {
        return m_L;
    }

    //! Get current stored inverse of L
    std::vector<float> getLinv() const
    {
        return {m_Linv[0], m_Linv[1], m_Linv[2]};
    }

    //! Get tilt factor xy
    float getTiltFactorXY() const
    {
        return m_xy;
    }

    //! Get tilt factor xz
    float getTiltFactorXZ() const
    {
        return m_xz;
    }

    //! Get tilt factor yz
    float getTiltFactorYZ() const
    {
        return m_yz;
    }

    //! Set tilt factor xy
    void setTiltFactorXY(float xy)
    {
        m_xy = xy;
    }

    //! Set tilt factor xz
    void setTiltFactorXZ(float xz)
    {
        m_xz = xz;
    }

    //! Set tilt factor yz
    void setTiltFactorYZ(float yz)
    {
        m_yz = yz;
    }

    //! Get the volume of the box (area in 2D)
    float getVolume() const
    {
        if (m_2d)
        {
            return m_L[0] * m_L[1];
        }
        return m_L[0] * m_L[1] * m_L[2];
    }

    xt::xarray<float> makeAbsolute(const xt::xarray<float>& f) const
    {
        auto&& v = xt::eval(m_lo + f * m_L);
        auto vx = xt::view(v, xt::all(), 0);
        auto vy = xt::view(v, xt::all(), 1);
        auto vz = xt::view(v, xt::all(), 2);
        vx += vy * m_xy + vz * m_xz;
        vy += vz * m_yz;
        if (m_2d)
        {
            xt::view(v, xt::all(), 2) = 0.0;
        }
        return v;
    }

    void makeAbsolute(const xt::xarray<float>& f, xt::xarray<float>& out) const
    {
        out = makeAbsolute(f);
    }

    void makeAbsolute(const vec3f& f, vec3f& out) const
    {
        out = makeAbsolute(f);
    }

    //! Convert a point's coordinate from absolute to fractional box coordinates.
    /*! \param v The vector of the point in absolute coordinates.
     *  \returns The vector of the point in fractional coordinates.
     */
     xt::xarray<float> makeFractional(const xt::xarray<float>& v) const
    {   
        auto&& delta = xt::eval(v - m_lo);
        auto vx = xt::view(v, xt::all(), 0);
        auto vy = xt::view(v, xt::all(), 1);
        auto vz = xt::view(v, xt::all(), 2);
        xt::view(delta, xt::all(), 0) -= (m_xz - m_yz * m_xy) * vz + m_xy * vy;
        xt::view(delta, xt::all(), 1) -= m_yz * vz;
        delta = delta / m_L;
        if (m_2d)
        {
            xt::view(delta, xt::all(), 2) = float(0.0);
        }
        return delta;
    }

    void makeFractional(const xt::xarray<float>& vecs, xt::xarray<float>& out) const
    {
        out = makeFractional(vecs);
    }

    void makeFractional(const vec3f& vec, vec3f& out) const
    {
        out = makeFractional(vec);
    }

    //! Get periodic image of a vector.
    /*! \param v The vector to check.
     *  \param image The image of a given point.
     */
    xt::xarray<int> getImage(const xt::xarray<float>& v) const
    {
        auto&& f = xt::eval(makeFractional(v) - 0.5);
        if (m_2d)
        {
            xt::view(f, xt::all(), 2) = float(0.0);
        }
        // image[0] = (int) ((f[0] >= float(0.0)) ? f[0] + float(0.5) : f[0] - float(0.5));
        // image[1] = (int) ((f[1] >= float(0.0)) ? f[1] + float(0.5) : f[1] - float(0.5));
        // image[2] = (int) ((f[2] >= float(0.0)) ? f[2] + float(0.5) : f[2] - float(0.5));
        auto fx = xt::view(f, xt::all(), 0);
        xt::where(fx >= 0.0, fx + 0.5, fx - 0.5);
        auto fy = xt::view(f, xt::all(), 1);
        xt::where(fy >= 0.0, fy + 0.5, fy - 0.5);
        auto fz = xt::view(f, xt::all(), 2);
        xt::where(fz >= 0.0, fz + 0.5, fz - 0.5);
        return xt::cast<int>(f);
    }

    //! Wrap a vector back into the box
    /*! \param v Vector to wrap, updated to the minimum image obeying the periodic settings
     *  \returns Wrapped vector
     */
    xt::xarray<float> wrap(const xt::xarray<float>& v) const
    {
        // Return quickly if the box is aperiodic
        if (!m_periodic[0] && !m_periodic[1] && !m_periodic[2])
        {
            return v;
        }

        auto v_frac = makeFractional(v);
        xt::xarray<bool> mask = xt::broadcast(m_periodic, {v_frac.shape()[0], m_periodic.shape()[0]});  // can't auto this...
        auto x = xt::fmod(xt::fmod(v_frac, 1.0) + 1.0, 1.0);
        auto m = xt::masked_view(v_frac, mask);
        m = x;
        return makeAbsolute(v_frac);
    }

    //! Wrap vectors back into the box in place
    /*! \param vecs Vectors to wrap, updated to the minimum image obeying the periodic settings
     *  \param Nvecs Number of vectors
     *  \param out The array in which to place the wrapped vectors.
     */
    // void wrap(const vec3f* vecs, unsigned int Nvecs, vec3f* out) const
    // {
    //     util::forLoopWrapper(0, Nvecs, [&](size_t begin, size_t end) {
    //         for (size_t i = begin; i < end; ++i)
    //         {
    //             out[i] = wrap(vecs[i]);
    //         }
    //     });
    // }

    //! Unwrap given positions to their absolute location in place
    /*! \param vecs Vectors of coordinates to unwrap
     *  \param images images flags for this point
        \param Nvecs Number of vectors
     *  \param out The array in which to place the wrapped vectors.
    */
    // void unwrap(const vec3f* vecs, const vec3i* images, unsigned int Nvecs, vec3f* out) const
    // {
    //     util::forLoopWrapper(0, Nvecs, [&](size_t begin, size_t end) {
    //         for (size_t i = begin; i < end; ++i)
    //         {
    //             out[i] = vecs[i] + getLatticeVector(0) * float(images[i][0])
    //                 + getLatticeVector(1) * float(images[i][1]);
    //             if (!m_2d)
    //             {
    //                 out[i] += getLatticeVector(2) * float(images[i][2]);
    //             }
    //         }
    //     });
    // }

    //! Compute center of mass for vectors
    /*! \param vecs Vectors to compute center of mass
     *  \param Nvecs Number of vectors
     *  \param masses Optional array of masses, of length Nvecs
     *  \return Center of mass as a vec3f
     */
    // vec3f centerOfMass(vec3f* vecs, size_t Nvecs, const float* masses = nullptr) const
    // {
    //     // This roughly follows the implementation in
    //     // https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
    //     float total_mass(0);
    //     vec3<std::complex<float>> xi_mean;

    //     for (size_t i = 0; i < Nvecs; ++i)
    //     {
    //         vec3f const phase(constants::TWO_PI * makeFractional(vecs[i]));
    //         vec3<std::complex<float>> const xi(std::polar(float(1.0), phase[0]),
    //                                            std::polar(float(1.0), phase[1]),
    //                                            std::polar(float(1.0), phase[2]));
    //         float const mass = (masses != nullptr) ? masses[i] : float(1.0);
    //         total_mass += mass;
    //         xi_mean += std::complex<float>(mass, 0) * xi;
    //     }
    //     xi_mean /= std::complex<float>(total_mass, 0);

    //     return wrap(makeAbsolute(vec3f(std::arg(xi_mean[0]), std::arg(xi_mean[1]), std::arg(xi_mean[2]))
    //                              / constants::TWO_PI));
    // }

    //! Subtract center of mass from vectors
    /*! \param vecs Vectors to center
     *  \param Nvecs Number of vectors
     *  \param masses Optional array of masses, of length Nvecs
     */
    // void center(vec3f* vecs, unsigned int Nvecs, const float* masses) const
    // {
    //     vec3f com(centerOfMass(vecs, Nvecs, masses));
    //     util::forLoopWrapper(0, Nvecs, [&](size_t begin, size_t end) {
    //         for (size_t i = begin; i < end; ++i)
    //         {
    //             vecs[i] = wrap(vecs[i] - com);
    //         }
    //     });
    // }

    //! Calculate distance between two points using boundary conditions
    /*! \param r_i Position of first point
        \param r_j Position of second point
    */
    // float computeDistance(const vec3f& r_i, const vec3f& r_j) const
    // {
    //     const vec3f r_ij = wrap(r_j - r_i);
    //     return std::sqrt(dot(r_ij, r_ij));
    // }

    //! Calculate distances between a set of query points and points.
    /*! \param query_points Query point positions.
        \param n_query_points The number of query points.
        \param points Point positions.
        \param n_points The number of points.
        \param distances Pointer to array of length n_query_points containing distances between each point and
       query_point (overwritten in place).
    */
    // void computeDistances(const vec3f* query_points, const unsigned int n_query_points,
    //                       const vec3f* points, float* distances) const
    // {
    //     util::forLoopWrapper(0, n_query_points, [&](size_t begin, size_t end) {
    //         for (size_t i = begin; i < end; ++i)
    //         {
    //             distances[i] = computeDistance(query_points[i], points[i]);
    //         }
    //     });
    // }

    //! Calculate all pairwise distances between a set of query points and points.
    /*! \param query_points Query point positions.
        \param n_query_points The number of query points.
        \param points Point positions.
        \param n_points The number of points.
        \param distances Pointer to array of length n_query_points*n_points containing distances between
       points and query_points (overwritten in place).
    */
    // void computeAllDistances(const vec3f* query_points, const unsigned int n_query_points,
    //                          const vec3f* points, const unsigned int n_points, float* distances) const
    // {
    //     util::forLoopWrapper2D(
    //         0, n_query_points, 0, n_points, [&](size_t begin_n, size_t end_n, size_t begin_m, size_t end_m) {
    //             for (size_t i = begin_n; i < end_n; ++i)
    //             {
    //                 for (size_t j = begin_m; j < end_m; ++j)
    //                 {
    //                     distances[i * n_points + j] = computeDistance(query_points[i], points[j]);
    //                 }
    //             }
    //         });
    // }

    //! Get mask of points that fit inside the box.
    /*! \param points Point positions.
        \param n_points The number of points.
        \param contains_mask Mask of points inside the box.
    */
    // void contains(vec3f* points, const unsigned int n_points, bool* contains_mask) const
    // {
    //     util::forLoopWrapper(0, n_points, [&](size_t begin, size_t end) {
    //         std::transform(&points[begin], &points[end], &contains_mask[begin],
    //                        [this](const vec3f& point) -> bool {
    //                            vec3i image{0, 0, 0};
    //                            getImage(point, image);
    //                            return image == vec3i{0, 0, 0};
    //                        });
    //     });
    // }

    //! Get the shortest distance between opposite boundary planes of the box
    /*! The distance between two planes of the lattice is 2 Pi/|b_i|, where
     *   b_1 is the reciprocal lattice vector of the Bravais lattice normal to
     *   the lattice vectors a_2 and a_3 etc.
     *
     * \return A vec3f containing the distance between the a_2-a_3, a_3-a_1 and
     *         a_1-a_2 planes for the triclinic lattice
     */
    // vec3f getNearestPlaneDistance() const
    // {
    //     vec3f dist;
    //     dist[0] = m_L[0] / std::sqrt(float(1.0) + m_xy * m_xy + (m_xy * m_yz - m_xz) * (m_xy * m_yz - m_xz));
    //     dist[1] = m_L[1] / std::sqrt(float(1.0) + m_yz * m_yz);
    //     dist[2] = m_L[2];

    //     return dist;
    // }

    /*! Get the lattice vector with index i
     *  \param i Index (0<=i<d) of the lattice vector, where d is dimension (2 or 3)
     *  \returns the lattice vector with index i
     */
    // vec3f getLatticeVector(unsigned int i) const
    // {
    //     if (i == 0)
    //     {
    //         return vec3f {m_L[0], 0.0, 0.0};
    //     }
    //     if (i == 1)
    //     {
    //         return vec3f {m_L[1] * m_xy, m_L[1], 0.0};
    //     }
    //     if (i == 2 && !m_2d)
    //     {
    //         return vec3f {m_L[2] * m_xz, m_L[2] * m_yz, m_L[2]};
    //     }
    //     throw std::out_of_range("Box lattice vector index requested does not exist.");
    // }

    //! Set the periodic flags
    /*! \param periodic Flags to set
     *  \post Period flags are set to \a periodic
     */
    void setPeriodic(bool x, bool y, bool z)
    {
        m_periodic = {x, y, z};
    }

    //! Set the periodic flag along x
    void setPeriodicX(bool x)
    {
        m_periodic[0] = x;
    }

    //! Set the periodic flag along y
    void setPeriodicY(bool y)
    {
        m_periodic[1] = y;
    }

    //! Set the periodic flag along z
    void setPeriodicZ(bool z)
    {
        m_periodic[2] = z;
    }

    //! Get the periodic flags
    vec3b getPeriodic() const
    {
        return {m_periodic[0], m_periodic[1], m_periodic[2]};
    }

    //! Get the periodic flag along x
    bool getPeriodicX() const
    {
        return m_periodic[0];
    }

    //! Get the periodic flag along y
    bool getPeriodicY() const
    {
        return m_periodic[1];
    }

    //! Get the periodic flag along z
    bool getPeriodicZ() const
    {
        return m_periodic[2];
    }

//     void enforce2D() const
//     {
//         if (!is2D())
//         {
//             throw std::invalid_argument("A 3D box was provided to a class that only supports 2D systems.");
//         }
//     }

//     void enforce3D() const
//     {
//         if (is2D())
//         {
//             throw std::invalid_argument("A 2D box was provided to a class that only supports 3D systems.");
//         }
//     }

float getXlo() const
{
    return m_lo[0];
}

float getYlo() const
{
    return m_lo[1];
}

float getZlo() const
{
    return m_lo[2];
}

float getXhi() const
{
    return m_hi[0];
}

float getYhi() const
{
    return m_hi[1];
}

float getZhi() const
{
    return m_hi[2];
}

private:
    vec3f m_lo;      //!< Minimum coords in the box
    vec3f m_hi;      //!< Maximum coords in the box
    vec3f m_L;       //!< L precomputed (used to avoid subtractions in boundary conditions)
    vec3f m_Linv;    //!< 1/L precomputed (used to avoid divisions in boundary conditions)
    float m_xy;            //!< xy tilt factor
    float m_xz;            //!< xz tilt factor
    float m_yz;            //!< yz tilt factor
    vec3b m_periodic; //!< 0/1 to determine if the box is periodic in each direction
    bool m_2d;             //!< Specify whether box is 2D.
};

}; }; // end namespace freud::box

#endif // BOX_H