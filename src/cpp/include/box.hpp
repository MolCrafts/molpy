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
#include <xtensor/xio.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

/*! \file Box.h
    \brief Represents simulation boxes and contains helpful wrapping functions.
*/

namespace molcpp {
namespace constants {
// Constant 2*pi for convenient use everywhere.
constexpr float TWO_PI = 2.0 * M_PI;
}; // namespace constants
}; // namespace molcpp

namespace molcpp {
namespace box {

//! Stores box dimensions and provides common routines for wrapping vectors back
//! into the box
/*! Box stores a standard HOOMD simulation box that goes from -L/2 to L/2 in
 each dimension, allowing Lx, Ly, Lz, and triclinic tilt factors xy, xz, and yz
 to be specified independently.
 *

    A number of utility functions are provided to work with coordinates in
 boxes. These are provided as inlined methods in the header file so they can be
 called in inner loops without sacrificing performance.
     - wrap()
     - unwrap()

    A Box can represent either a two or three dimensional box. By default, a Box
 is 3D, but can be set as 2D with the method set2D(), or via an optional boolean
 argument to the constructor. is2D() queries if a Box is 2D or not. 2D boxes
 have a "volume" of Lx * Ly, and Lz is set to 0. To keep programming simple, all
 inputs and outputs are still 3-component vectors even for 2D boxes. The third
 component ignored (assumed set to 0).
*/
class Box {
public:
  //! Nullary constructor for Cython
  Box() {
    _matrix = mat3f{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    _inv_matrix = mat3f{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    _periodic = {true, true, true};
    _origin = {0, 0, 0};
  }

  //! Construct a square/cubic box
  explicit Box(float L, bool _2d = false) {
    setL(L, L, L);
    _periodic = {true, true, true};
    _origin = {0, 0, 0};
  }

  //! Construct an orthorhombic box
  explicit Box(float Lx, float Ly, float Lz) {
    setL(Lx, Ly, Lz);
    _periodic = {true, true, true};
    _origin = {0, 0, 0};
  }

  //! Construct a triclinic box
  explicit Box(float Lx, float Ly, float Lz, float xy, float xz, float yz,
               bool _2d = false) {
    _matrix = mat3f{{Lx, xy, xz}, {0, Ly, yz}, {0, 0, Lz}};
    _inv_matrix = xt::linalg::inv(_matrix);
    _periodic = {true, true, true};
  }

  bool operator==(const Box &b) const {
    return xt::allclose(this->_matrix, b._matrix) && xt::allclose(this->_origin, b._origin) &&
           this->_periodic == b._periodic;
  }

  bool operator!=(const Box &b) const { return !(*this == b); }

  //! Set L, box lengths, inverses.  Box is also centered at zero.
  void setL(const float Lx, const float Ly, const float Lz) {
    _matrix = mat3f{{Lx, 0, 0}, {0, Ly, 0}, {0, 0, Lz}};
    _inv_matrix = xt::linalg::inv(_matrix);
  }

  //! Set whether box is 2D
  void set2D(bool _2d) {
    _matrix = mat3f{
        {_matrix[0], _matrix[1], 0}, {_matrix[3], _matrix[4], 0}, {0, 0, 0}};
    _inv_matrix = xt::linalg::inv(_matrix);
  }

  //! Set the origin of the box
  void setOrigin(const float x, const float y, const float z) {
    _origin = {x, y, z};
  }

  //! Set the origin of the box
  void setOrigin(const vec3f origin) {
    _origin = origin;
  }

  //! Returns whether box is two dimensional
  bool is2D() const { return _matrix(2, 2) == 0; }

  //! Get the value of Lx
  float getLx() const { return _matrix(0, 0); }

  //! Get the value of Ly
  float getLy() const { return _matrix(1, 1); }

  //! Get the value of Lz
  float getLz() const { return _matrix(2, 2); }

  //! Get current L
  vec3f getL() const { return xt::diag(_matrix); }

  //! Get current stored inverse of L
  vec3f getLinv() const { return xt::diag(_inv_matrix); }

  //! Get tilt factor xy
  float getTiltFactorXY() const { return _matrix(0, 1); }

  //! Get tilt factor xz
  float getTiltFactorXZ() const { return _matrix(0, 2); }

  //! Get tilt factor yz
  float getTiltFactorYZ() const { return _matrix(1, 2); }

  //! Set tilt factor xy
  void setTiltFactorXY(float xy) { _matrix(0, 1) = xy; }

  //! Set tilt factor xz
  void setTiltFactorXZ(float xz) { _matrix(0, 2) = xz; }

  //! Set tilt factor yz
  void setTiltFactorYZ(float yz) { _matrix(1, 2) = yz; }

  //! Get the volume of the box (0 in 2D)
  float getVolume() const {
    return _matrix(0, 0) * _matrix(1, 1) * _matrix(2, 2);
  }

  //! Convert fractional coordinates into absolute coordinates
  /*! \param f Fractional coordinates between 0 and 1 within
   *         parallelepipedal box
   *  \return A vector inside the box corresponding to f
   */
  xt::xarray<float> makeAbsolute(const xt::xarray<float> &f) const {
    auto v = xt::linalg::dot(f, _matrix);
    std::cout << f << std::endl;
    std::cout << _matrix << std::endl;
    std::cout << _inv_matrix << std::endl;
    std::cout << v << std::endl;
    return v;
  }

  void makeAbsolute(const xt::xarray<float> &f, xt::xarray<float> &out) const {
    out = makeAbsolute(f);
  }

  void makeAbsolute(const vec3f &f, vec3f &out) const { out = makeAbsolute(f); }

  //! Convert a point's coordinate from absolute to fractional box coordinates.
  /*! \param v The vector of the point in absolute coordinates.
   *  \returns The vector of the point in fractional coordinates.
   */
  // xt::xarray<float> makeFractional(const xt::xarray<float> &v) const {
  //   auto &&delta = xt::eval(v - m_lo);
  //   auto vx = xt::view(v, xt::all(), 0);
  //   auto vy = xt::view(v, xt::all(), 1);
  //   auto vz = xt::view(v, xt::all(), 2);
  //   xt::view(delta, xt::all(), 0) -= (m_xz - m_yz * m_xy) * vz + m_xy * vy;
  //   xt::view(delta, xt::all(), 1) -= m_yz * vz;
  //   delta = delta / m_L;
  //   if (m_2d) {
  //     xt::view(delta, xt::all(), 2) = float(0.0);
  //   }
  //   return delta;
  // }

  // void makeFractional(const xt::xarray<float> &vecs,
  //                     xt::xarray<float> &out) const {
  //   out = makeFractional(vecs);
  // }

  // void makeFractional(const vec3f &vec, vec3f &out) const {
  //   out = makeFractional(vec);
  // }

  //! Get periodic image of a vector.
  /*! \param v The vector to check.
   *  \param image The image of a given point.
   */
  // xt::xarray<int> getImage(const xt::xarray<float> &v) const {
  //   auto &&f = xt::eval(makeFractional(v) - 0.5);
  //   if (m_2d) {
  //     xt::view(f, xt::all(), 2) = float(0.0);
  //   }
  //   auto fx = xt::view(f, xt::all(), 0);
  //   xt::where(fx >= 0.0, fx + 0.5, fx - 0.5);
  //   auto fy = xt::view(f, xt::all(), 1);
  //   xt::where(fy >= 0.0, fy + 0.5, fy - 0.5);
  //   auto fz = xt::view(f, xt::all(), 2);
  //   xt::where(fz >= 0.0, fz + 0.5, fz - 0.5);
  //   return xt::cast<int>(f);
  // }

  //! Wrap a vector back into the box
  /*! \param v Vector to wrap, updated to the minimum image obeying the periodic
   * settings \returns Wrapped vector
   */
  // xt::xarray<float> wrap(const xt::xarray<float> &v) const {
  //   // Return quickly if the box is aperiodic
  //   if (!m_periodic[0] && !m_periodic[1] && !m_periodic[2]) {
  //     return v;
  //   }

  //   auto v_frac = makeFractional(v);
  //   xt::xarray<bool> mask = xt::broadcast(
  //       m_periodic,
  //       {v_frac.shape()[0], m_periodic.shape()[0]}); // can't auto this...
  //   auto x = xt::fmod(xt::fmod(v_frac, 1.0) + 1.0, 1.0);
  //   auto m = xt::masked_view(v_frac, mask);
  //   m = x;
  //   return makeAbsolute(v_frac);
  // }

  mat3f getMatrix() const {
    return _matrix;
  }

  mat3f getInvMatrix() const {
    return _inv_matrix;
  }

  //! Unwrap given positions to their absolute location in place
  /*! \param vecs Vectors of coordinates to unwrap
   *  \param images images flags for this point
      \param Nvecs Number of vectors
   *  \param out The array in which to place the wrapped vectors.
  */
  // xt::xarray<float> unwrap(const xt::xarray<float> &v,
  //                          const xt::xarray<int> images) const {
  //   auto mat = getMatrix();
  //   auto unwrapped = v + mat * xt::transpose(images);
  //   return unwrapped;
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
      \param distances Pointer to array of length n_query_points containing
     distances between each point and query_point (overwritten in place).
  */
  // void computeDistances(const vec3f* query_points, const unsigned int
  // n_query_points,
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
      \param distances Pointer to array of length n_query_points*n_points
     containing distances between points and query_points (overwritten in
     place).
  */
  // void computeAllDistances(const vec3f* query_points, const unsigned int
  // n_query_points,
  //                          const vec3f* points, const unsigned int n_points,
  //                          float* distances) const
  // {
  //     util::forLoopWrapper2D(
  //         0, n_query_points, 0, n_points, [&](size_t begin_n, size_t end_n,
  //         size_t begin_m, size_t end_m) {
  //             for (size_t i = begin_n; i < end_n; ++i)
  //             {
  //                 for (size_t j = begin_m; j < end_m; ++j)
  //                 {
  //                     distances[i * n_points + j] =
  //                     computeDistance(query_points[i], points[j]);
  //                 }
  //             }
  //         });
  // }

  //! Get mask of points that fit inside the box.
  /*! \param points Point positions.
      \param n_points The number of points.
      \param contains_mask Mask of points inside the box.
  */
  // void contains(vec3f* points, const unsigned int n_points, bool*
  // contains_mask) const
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
  //     dist[0] = m_L[0] / std::sqrt(float(1.0) + m_xy * m_xy + (m_xy * m_yz -
  //     m_xz) * (m_xy * m_yz - m_xz)); dist[1] = m_L[1] / std::sqrt(float(1.0)
  //     + m_yz * m_yz); dist[2] = m_L[2];

  //     return dist;
  // }

  /*! Get the lattice vector with index i
   *  \param i Index (0<=i<d) of the lattice vector, where d is dimension (2 or
   * 3) \returns the lattice vector with index i
   */
  vec3f getLatticeVector(unsigned int i) const {
    return xt::view(_matrix, xt::all(), i);
  }

  //! Set the periodic flags
  /*! \param periodic Flags to set
   *  \post Period flags are set to \a periodic
   */
  void setPeriodic(bool x, bool y, bool z) { _periodic = {x, y, z}; }

  //! Set the periodic flag along x
  void setPeriodicX(bool x) { _periodic[0] = x; }

  //! Set the periodic flag along y
  void setPeriodicY(bool y) { _periodic[1] = y; }

  //! Set the periodic flag along z
  void setPeriodicZ(bool z) { _periodic[2] = z; }

  //! Get the periodic flags
  vec3b getPeriodic() const {
    return _periodic;
  }

  //! Get the periodic flag along x
  bool getPeriodicX() const { return _periodic[0]; }

  //! Get the periodic flag along y
  bool getPeriodicY() const { return _periodic[1]; }

  //! Get the periodic flag along z
  bool getPeriodicZ() const { return _periodic[2]; }

  float getXlo() const { return _origin(0); }

  float getYlo() const { return _origin(1); }

  float getZlo() const { return _origin(2); }

  float getXhi() const { return _matrix(0, 0) + _origin(0); }

  float getYhi() const { return _matrix(1, 1) + _origin(1); }

  float getZhi() const { return _matrix(2, 2) + _origin(2); }

private:
  mat3f _matrix;
  mat3f _inv_matrix;
  vec3b _periodic; //!< 0/1 to determine if the box is periodic in each direction
  vec3f _origin;   //!< Origin of the box
};

}; // namespace box
}; // namespace molcpp

#endif // BOX_H