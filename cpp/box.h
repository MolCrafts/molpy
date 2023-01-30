// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef BOX_H
#define BOX_H

#include "vectorMath.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/*! \file Box.h
    \brief Represents simulation boxes and contains helpful wrapping functions.
*/

namespace molpy
{

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
            m_periodic = vec3<bool>(true, true, true);
            m_xy = m_xz = m_yz = 0;
        }

        //! Construct a square/cubic box
        explicit Box(float L, bool _2d = false)
        {
            m_2d = _2d; // Assign before calling setL!
            setL(L, L, L);
            m_periodic = vec3<bool>(true, true, true);
            m_xy = m_xz = m_yz = 0;
        }

        //! Construct an orthorhombic box
        Box(float Lx, float Ly, float Lz, bool _2d = false)
        {
            m_2d = _2d; // Assign before calling setL!
            setL(Lx, Ly, Lz);
            m_periodic = vec3<bool>(true, true, true);
            m_xy = m_xz = m_yz = 0;
        }

        //! Construct a triclinic box
        Box(float Lx, float Ly, float Lz, float xy, float xz, float yz, bool _2d = false)
        {
            m_2d = _2d; // Assign before calling setL!
            setL(Lx, Ly, Lz);
            m_periodic = vec3<bool>(true, true, true);
            m_xy = xy;
            m_xz = xz;
            m_yz = yz;
        }

        inline bool operator==(const Box &b) const
        {
            return ((this->getL() == b.getL()) && (this->getTiltFactorXY() == b.getTiltFactorXY()) && (this->getTiltFactorXZ() == b.getTiltFactorXZ()) && (this->getTiltFactorYZ() == b.getTiltFactorYZ()) && (this->getPeriodicX() == b.getPeriodicX()) && (this->getPeriodicY() == b.getPeriodicY()) && (this->getPeriodicZ() == b.getPeriodicZ()));
        }

        inline bool operator!=(const Box &b) const
        {
            return !(*this == b);
        }

        //! Set L, box lengths, inverses.  Box is also centered at zero.
        void setL(const vec3<float> &L)
        {
            setL(L.x, L.y, L.z);
        }

        //! Set L, box lengths, inverses.  Box is also centered at zero.
        void setL(const float Lx, const float Ly, const float Lz)
        {
            if (m_2d)
            {
                m_L = vec3<float>(Lx, Ly, 0);
                m_Linv = vec3<float>(1 / m_L.x, 1 / m_L.y, 0);
            }
            else
            {
                m_L = vec3<float>(Lx, Ly, Lz);
                m_Linv = vec3<float>(1 / m_L.x, 1 / m_L.y, 1 / m_L.z);
            }

            m_hi = m_L / float(2.0);
            m_lo = -m_hi;
        }

        //! Set whether box is 2D
        void set2D(bool _2d)
        {
            m_2d = _2d;
            m_L.z = 0;
            m_Linv.z = 0;
        }

        //! Returns whether box is two dimensional
        bool is2D() const
        {
            return m_2d;
        }

        //! Get the value of Lx
        float getLx() const
        {
            return m_L.x;
        }

        //! Get the value of Ly
        float getLy() const
        {
            return m_L.y;
        }

        //! Get the value of Lz
        float getLz() const
        {
            return m_L.z;
        }

        //! Get current L
        vec3<float> getL() const
        {
            return m_L;
        }

        //! Get current stored inverse of L
        vec3<float> getLinv() const
        {
            return m_Linv;
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
                return m_L.x * m_L.y;
            }
            return m_L.x * m_L.y * m_L.z;
        }

        //! Set the periodic flags
        /*! \param periodic Flags to set
         *  \post Period flags are set to \a periodic
         */
        void setPeriodic(vec3<bool> periodic)
        {
            m_periodic = periodic;
        }

        void setPeriodic(bool x, bool y, bool z)
        {
            m_periodic = vec3<bool>(x, y, z);
        }

        //! Set the periodic flag along x
        void setPeriodicX(bool x)
        {
            m_periodic = vec3<bool>(x, m_periodic.y, m_periodic.z);
        }

        //! Set the periodic flag along y
        void setPeriodicY(bool y)
        {
            m_periodic = vec3<bool>(m_periodic.x, y, m_periodic.z);
        }

        //! Set the periodic flag along z
        void setPeriodicZ(bool z)
        {
            m_periodic = vec3<bool>(m_periodic.x, m_periodic.y, z);
        }

        //! Get the periodic flags
        vec3<bool> getPeriodic() const
        {
            return {m_periodic.x, m_periodic.y, m_periodic.z};
        }

        //! Get the periodic flag along x
        bool getPeriodicX() const
        {
            return m_periodic.x;
        }

        //! Get the periodic flag along y
        bool getPeriodicY() const
        {
            return m_periodic.y;
        }

        //! Get the periodic flag along z
        bool getPeriodicZ() const
        {
            return m_periodic.z;
        }

        //! Convert fractional coordinates into absolute coordinates
        /*! \param f Fractional coordinates between 0 and 1 within
         *         parallelepipedal box
         *  \return A vector inside the box corresponding to f
         */
        py::array makeAbsolute(const py::array_t<double, py::array::c_style | py::array::forcecast> &f) const
        {
            if (f.ndim == 1)
                auto r = f.unchecked<1>();
                auto result = py::array_t<double>(3);
                auto result_buf = result.request();
                auto result_ptr = (double *)result_buf.ptr;
                result_ptr[0] = m_lo.x + r(0) * m_L.x;
                result_ptr[1] = m_lo.y + r(1) * m_L.y;
                result_ptr[2] = m_lo.z + r(2) * m_L.z;
                return result;

            else if (f.ndim == 2)
                auto r = f.unchecked<2>();
                auto result = py::array_t<double>(f.shape(0), f.shape(1));
                auto result_buf = result.request();
                auto result_ptr = (double *)result_buf.ptr;
                for (int i = 0; i < f.shape(0); i++)
                {
                    result_ptr[i * 3] = m_lo + f(i, 0) * m_L;
                    result_ptr[i * 3 + 1] = m_lo + f(i, 1) * m_L;
                    result_ptr[i * 3 + 2] = m_lo + f(i, 2) * m_L;
                }
                return result;
            else
                throw std::runtime_error("makeAbsolute: array must be 1D or 2D");

        }

    private:
        vec3<float> m_lo;      //!< Minimum coords in the box
        vec3<float> m_hi;      //!< Maximum coords in the box
        vec3<float> m_L;       //!< L precomputed (used to avoid subtractions in boundary conditions)
        vec3<float> m_Linv;    //!< 1/L precomputed (used to avoid divisions in boundary conditions)
        float m_xy;            //!< xy tilt factor
        float m_xz;            //!< xz tilt factor
        float m_yz;            //!< yz tilt factor
        vec3<bool> m_periodic; //!< 0/1 to determine if the box is periodic in each direction
        bool m_2d;             //!< Specify whether box is 2D.
    }
}

#endif // BOX_H