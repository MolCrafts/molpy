#include <catch2/catch_test_macros.hpp>
#include "types.hpp"
#include <xtensor/xfixed.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

TEST_CASE( "Test Vec3 Init" ) {

    const vec3b v = {true, false, true};
    REQUIRE( v[0] == true );

    const vec3f x {1.0, 2.0, 3.0};
    REQUIRE( x[0] == 1.0 );

}

TEST_CASE( "Test Vec3 Inplace" ) {

    vec3f v = {1.0, 2.0, 3.0};
    v += 3;
    REQUIRE( v[0] == 4.0 );

    vec3f x = {1.0, 2.0, 3.0};
    vec3f y = {1.0, 2.0, 3.0};
    x += y;
    REQUIRE( x[0] == 2.0 );

}

TEST_CASE( "Test Xtensor View" ) {

    xt::xarray<float> z = {{1.0, 2.0, 3.0}, {2.0, 3.0, 4.0}};
    auto zx = xt::view(z, xt::all(), 0);
    zx += 1;
    REQUIRE( z[{0, 0}] == 2.0 );
    auto zy = xt::view(z, xt::all(), 1);
    zx += zy * 2;
    REQUIRE( z[{0, 0}] == 6.0 );
    

}