#pragma once

#include <xtensor/xfixed.hpp>

using vec3f = xt::xtensor_fixed<float, xt::xshape<3>>;
using vec3b = xt::xtensor_fixed<bool, xt::xshape<3>>;
using vec3i = xt::xtensor_fixed<int, xt::xshape<3>>;

using mat3f = xt::xtensor_fixed<float, xt::xshape<3, 3>>;
