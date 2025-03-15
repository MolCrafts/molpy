#include <catch2/catch_test_macros.hpp>
#include "box.hpp"
#include <xtensor/xfixed.hpp>
#include <xtensor/xarray.hpp>

using molcpp::box::Box;

TEST_CASE( "Test Box Init" ) {
    Box box1;
    Box box2(1.0, 2.0, 3.0);
    Box box3 {1.0, 2.0, 3.0};
}

SCENARIO( "Test Absolute Coordinates" ) {

    Box box(2.0, 2.0, 2.0);

    GIVEN ( "a vec3f" ) {
        vec3f f_point = {0.5, 0.25, 0.75};
        vec3f point = {0, -0.5, 0.5};

        THEN( "test absolute" ) {
            auto test_coordinates = box.makeAbsolute(f_point);
            REQUIRE( xt::allclose(test_coordinates, point) );
            
            // test out of place operation
            REQUIRE( xt::allclose(f_point, vec3f{0.5, 0.25, 0.75}) );
        }

    GIVEN ( "a xt::xarray" ) {

        
        xt::xarray<float> f_point = {{0.5, 0.25, 0.75}, {0, 0, 0}, {0.5, 0.5, 0.5}};
        xt::xarray<float> point = {{0, -0.5, 0.5}, {-1, -1, -1}, {0, 0, 0}};
    }

        THEN( "test absolute" ) {

        auto test_coordinates = box.makeAbsolute(f_point);
        REQUIRE( xt::allclose(test_coordinates, point) );
    }
    }

    GIVEN( "a vec3f and out" ) {
        vec3f f_point = {0.5, 0.25, 0.75};
        vec3f point = {0, -0.5, 0.5};

        THEN( "test absolute" ) {
            vec3f out = {0, 0, 0};
            box.makeAbsolute(f_point, out);
            REQUIRE( xt::allclose(out, point) );
        }
    }

    GIVEN( "a xarray and out" ) {
        xt::xarray<float> f_point = {0.5, 0.25, 0.75};
        xt::xarray<float> point = {0, -0.5, 0.5};

        THEN( "test absolute" ) {
            xt::xarray<float> out = {0, 0, 0};
            box.makeAbsolute(f_point, out);
            REQUIRE( xt::allclose(out, point) );
        }
    }

}

SCENARIO( "Test Fractional Coordinates" ) {

    Box box(2.0, 2.0, 2.0);

    GIVEN ( "a vec3f" ) {
        vec3f f_point = {5.5, 0.25, 0.75};
        vec3f point = {10, -0.5, 0.5};

        THEN( "test fractional" ) {
            auto test_coordinates = box.makeFractional(point);
            REQUIRE( xt::allclose(test_coordinates, f_point) );
            
            // test out of place operation
            REQUIRE( xt::allclose(point, vec3f{10, -0.5, 0.5}) );
        }
    }

    GIVEN ( "triclinic") {

        Box tbox(2.0, 2.0, 2.0, 1.0, 0.0, 0.0);
        xt::xarray<float> f_point = {{10, -5, -5}, {0, 0.5, 0}, {1.2, 6.4, 8.4}};
        xt::xarray<float> point = {{8, -2, -2}, {0.25, 0.75, 0.5}, {-2.1, 3.7, 4.7}};

        THEN( "test fractional" ) {
            auto test_coordinates = tbox.makeFractional(f_point);
            REQUIRE( xt::allclose(test_coordinates, point) );
        }
    }

    GIVEN ( "a xt::xarray" ) {
        xt::xarray<float> f_point = {{0.5, 0.25, 0.75}, {0, 0, 0}, {0.5, 0.5, 0.5}};
        xt::xarray<float> point = {{0, -0.5, 0.5}, {-1, -1, -1}, {0, 0, 0}};

        THEN( "test fractional" ) {
            auto test_coordinates = box.makeFractional(point);
            REQUIRE( xt::allclose(test_coordinates, f_point) );
        }
    }

    GIVEN( "a vec3f and out" ) {
        vec3f f_point = {0.5, 0.25, 0.75};
        vec3f point = {0, -0.5, 0.5};

        THEN( "test fractional" ) {
            vec3f out = {0, 0, 0};
            box.makeFractional(point, out);
            REQUIRE( xt::allclose(out, f_point) );
        }
    }

    GIVEN( "a xarray and out" ) {
        xt::xarray<float> point = {0, -0.5, 0.5};
        xt::xarray<float> f_point = {0.5, 0.25, 0.75};

        THEN( "test fractional" ) {
            xt::xarray<float> out = {0, 0, 0};
            box.makeFractional(point, out);
            REQUIRE( xt::allclose(out, f_point
            ) );    
        }
    }

    GIVEN( "triclinic" ) {
        Box box(2.0, 2.0, 2.0, 1.0, 0, 0);
        vec3f f_point = {8, -2, -2};
        vec3f point = {10, -5, -5};

        THEN( "test fractional" ) {
            auto test_coordinates = box.makeFractional(point);
            REQUIRE( xt::allclose(test_coordinates, f_point) );
        }
    }
}

SCENARIO( "Test Get Image" ) {

    GIVEN( "3d system" ) {
        Box box(2.0, 2.0, 2.0);
        xt::xarray<float> points = {{50, 40, 30}, {-10, 0, 0}};
        xt::xarray<int> image = {{25, 20, 15}, {-5, 0, 0}};

        THEN( "test get image" ) {
            auto test_image = box.getImage(points);
            REQUIRE( xt::allclose(test_image, image) );
        }
    }

    GIVEN( "2d system" ) {
        Box box(2.0, 2.0, 0.0, true);
        xt::xarray<float> points = {{50, 40, 0}, {-10, 0, 0}};
        xt::xarray<int> image = {{25, 20, 0}, {-5, 0, 0}};

        THEN( "test get image" ) {
            auto test_image = box.getImage(points);
            REQUIRE( xt::allclose(test_image, image) );
        }
        
    }

}

SCENARIO( "Test Wrap" ) {

    // GIVEN(" triclinic ") {
    //     Box box(2.0, 2.0, 2.0, 1.0, 0, 0);
    //     xt::xarray<float> points = {{10, -5, -5}, {0, 0.5, 0}, {1.2, 6.4, 8.4}};
    //     xt::xarray<float> wrapped = {{-2, -1, -1}, {0, 0.5, 0}, {1.2, 0.4, 0.4}};

    //     THEN( "test wrap" ) {
    //         auto test_wrap = box.wrap(points);
    //         std::cout << test_wrap << std::endl;
    //         REQUIRE( xt::allclose(test_wrap, wrapped) );
    //     }
    // }

    GIVEN( "partial periodic" ) {
        Box box(2.0, 2.0, 2.0, 1.0, 0, 0);
        box.setPeriodic(false, true, true);
        xt::xarray<float> points = {{10, -5, -5}, {0, 0.5, 0}};
        xt::xarray<float> wrapped = {{14, -1, -1}, {0, 0.5, 0}};

        THEN( "test wrap" ) {
            auto test_wrap = box.wrap(points);
            REQUIRE( xt::allclose(test_wrap, wrapped) );
        }
    }
}